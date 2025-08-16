# app/services/posture_ml.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ["param1", "param2", "param3"]

class PostureModels:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.rf_path = self.base_dir / "rf_classifier.joblib"
        self.km_path = self.base_dir / "kmeans.joblib"
        self.iso_path = self.base_dir / "isoforest.joblib"
        self.scaler_path = self.base_dir / "scaler.joblib"

    def fit_all(self, df: pd.DataFrame) -> dict:
        report = {}
        X = df[FEATURE_COLS].astype(float).values

        # Scale for KMeans/IsolationForest; RF is scale-insensitive but we’ll still use raw X there.
        scaler = StandardScaler().fit(X)
        joblib.dump(scaler, self.scaler_path)

        # 1) Supervised: RandomForestClassifier (Good/Bad)
        if "label" in df.columns and df["label"].notna().any():
            mask = df["label"].notna()
            X_lab = df.loc[mask, FEATURE_COLS].astype(float).values
            y = df.loc[mask, "label"].astype(int).values
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=None, random_state=42, n_jobs=-1, class_weight="balanced"
            )
            # cross-validated accuracy
            scores = cross_val_score(rf, X_lab, y, cv=5, scoring="accuracy")
            rf.fit(X_lab, y)
            joblib.dump(rf, self.rf_path)
            report["rf_cv_accuracy"] = float(scores.mean())
            report["rf_cv_std"] = float(scores.std())
        else:
            report["rf_cv_accuracy"] = None
            report["rf_cv_std"] = None

        # 2) Unsupervised clusters: KMeans
        Xs = scaler.transform(X)
        # choose a small k that’s interpretable
        km = KMeans(n_clusters=4, n_init="auto", random_state=42)
        km.fit(Xs)
        joblib.dump(km, self.km_path)
        report["kmeans_inertia"] = float(km.inertia_)

        # 3) Anomaly detection: IsolationForest
        iso = IsolationForest(
            n_estimators=300, contamination="auto", random_state=42
        ).fit(Xs)
        joblib.dump(iso, self.iso_path)
        report["isoforest_fitted"] = True

        return report

    def predict_all(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()

        # Load models if present
        scaler = joblib.load(self.scaler_path)
        Xs = scaler.transform(out[FEATURE_COLS].astype(float).values)

        # RandomForest predictions if available
        if Path(self.rf_path).exists():
            rf = joblib.load(self.rf_path)
            proba = rf.predict_proba(out[FEATURE_COLS].astype(float).values)[:, 1]
            out["ml_good_prob"] = proba  # probability posture is good
            out["ml_good_pred"] = (proba >= 0.5).astype(int)
        else:
            out["ml_good_prob"] = None
            out["ml_good_pred"] = None

        # KMeans clusters
        if Path(self.km_path).exists():
            km = joblib.load(self.km_path)
            out["cluster"] = km.predict(Xs)
        else:
            out["cluster"] = None

        # Anomaly scores (lower score => more anomalous); convert to “anomaly” boolean
        if Path(self.iso_path).exists():
            iso = joblib.load(self.iso_path)
            score = iso.score_samples(Xs)
            out["anomaly_score"] = score
            # Flag the bottom 5% as anomalies
            threshold = np.quantile(score, 0.05)
            out["is_anomaly"] = (score <= threshold).astype(int)
        else:
            out["anomaly_score"] = None
            out["is_anomaly"] = None

        return out

def suggestions_from_analytics(user_daily, user_df, peak_hours, longest_streak):
    suggestions = []
    # Trend-based
    if not user_daily.empty and user_daily["good_rate"].notna().any():
        last7 = user_daily.tail(7)
        if len(last7) >= 3:
            trend = last7["good_rate"].mean() - user_daily["good_rate"].head(7).mean()
            if pd.notna(trend):
                if trend < -0.05:
                    suggestions.append("Your good-posture rate is trending down in the last week. Consider shorter, more frequent breaks.")
                elif trend > 0.05:
                    suggestions.append("Great! Your good-posture rate improved this week — keep the routine that’s working.")

    # Time-of-day pattern
    if peak_hours:
        hrs = ", ".join(f"{h['hour']:02d}:00" for h in peak_hours)
        suggestions.append(f"Most ‘bad posture’ episodes occur around: {hrs}. Try scheduling posture checks at those times.")

    # Anomalies
    if not user_df.empty and "is_anomaly" in user_df.columns:
        spikes = int(user_df["is_anomaly"].sum())
        if spikes >= 3:
            suggestions.append("Detected several unusual posture spikes. Verify sensor placement and consider a quick stretch routine.")

    # Streak motivation
    if longest_streak >= 3:
        suggestions.append(f"You kept a good-posture streak for {longest_streak} days — set a new target to beat it!")

    if not suggestions:
        suggestions.append("Consistency looks stable. Keep brief micro-breaks every 30–45 minutes and re-center your posture.")
    return suggestions
