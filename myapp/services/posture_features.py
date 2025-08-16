# app/services/posture_features.py
from django.db.models import Q
from django.utils import timezone
import pandas as pd

FEATURE_COLS = ["param1", "param2", "param3"]

def queryset_to_df(qs):
    # Only keep rows that have all three params
    rows = list(qs.values("id", "user_id", "param1", "param2", "param3", "timestamp", "prediction"))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Normalize label: Good/Bad → 1/0 (if available)
    if "prediction" in df.columns:
        df["label"] = df["prediction"].fillna("").str.lower().map({"good": 1, "bad": 0})
    # Timestamps
    df["ts"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["ts"].dt.date
    df["dow"] = df["ts"].dt.dayofweek  # 0=Mon
    df["hour"] = df["ts"].dt.hour
    # Basic guards
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = None
    df = df.dropna(subset=FEATURE_COLS, how="any")
    return df

def daily_summary(df: pd.DataFrame):
    if df.empty: 
        return pd.DataFrame()
    # Daily good-rate (if labels exist)
    has_label = df["label"].notna().any()
    grp = df.groupby("date")
    out = pd.DataFrame()
    out["count"] = grp["id"].count()
    if has_label:
        out["good_rate"] = grp["label"].mean()
    else:
        out["good_rate"] = None
    out["avg_param1"] = grp["param1"].mean()
    out["avg_param2"] = grp["param2"].mean()
    out["avg_param3"] = grp["param3"].mean()
    return out.reset_index()

def weekly_summary(df: pd.DataFrame):
    if df.empty: 
        return pd.DataFrame()
    df = df.copy()
    df["week"] = df["ts"].dt.to_period("W").apply(lambda r: r.start_time.date())
    grp = df.groupby("week")
    out = pd.DataFrame()
    out["count"] = grp["id"].count()
    if df["label"].notna().any():
        out["good_rate"] = grp["label"].mean()
    else:
        out["good_rate"] = None
    out["avg_param1"] = grp["param1"].mean()
    out["avg_param2"] = grp["param2"].mean()
    out["avg_param3"] = grp["param3"].mean()
    return out.reset_index()

def peak_bad_hours(df: pd.DataFrame):
    if df.empty or df["label"].isna().all():
        return []
    bad = df[df["label"] == 0]
    if bad.empty:
        return []
    freq = bad.groupby("hour")["id"].count().sort_values(ascending=False)
    return [{"hour": int(h), "count": int(c)} for h, c in freq.head(3).items()]

def longest_good_streak(df: pd.DataFrame):
    if df.empty or df["label"].isna().all():
        return 0
    df_sorted = df.sort_values("ts")
    streak = best = 0
    prev_date = None
    for day, g in df_sorted.groupby("date"):
        day_good = g["label"].mean()  # proportion
        if day_good >= 0.7:  # threshold for “mostly good”
            if prev_date is None or (day - prev_date).days == 1:
                streak += 1
            else:
                streak = 1
            best = max(best, streak)
            prev_date = day
        else:
            prev_date = day
            streak = 0
    return int(best)
def suggestions_from_analytics(daily_df, pred_df, peak_bad_hours_list, longest_streak):
    """
    Generate posture suggestions based on analytics.

    Args:
        daily_df (pd.DataFrame): Daily summary dataframe
        pred_df (pd.DataFrame): Full predicted dataframe
        peak_bad_hours_list (list): List of dicts with peak bad hours
        longest_streak (int): Longest streak of good posture

    Returns:
        list: List of suggestion strings
    """
    suggestions = []

    # Check overall daily good rate
    if not daily_df.empty and "good_rate" in daily_df.columns:
        avg_good = daily_df["good_rate"].mean()
        if avg_good >= 0.8:
            suggestions.append("Great job! Your posture is mostly good.")
        elif avg_good >= 0.5:
            suggestions.append("Your posture is okay, try to sit straighter when possible.")
        else:
            suggestions.append("Your posture needs improvement. Consider posture exercises.")

    # Peak bad hours
    if peak_bad_hours_list:
        hours = ", ".join(str(h["hour"]) + "h" for h in peak_bad_hours_list)
        suggestions.append(f"You often slouch during these hours: {hours}. Take short breaks and adjust your posture.")

    # Longest streak
    if longest_streak < 3:
        suggestions.append("Try to maintain a good posture streak longer than 3 days for better results.")
    else:
        suggestions.append(f"Your longest good posture streak is {longest_streak} days. Keep it up!")

    # Optional: add tips based on param averages
    if not pred_df.empty:
        avg_params = pred_df[["param1", "param2", "param3"]].mean()
        if (avg_params > 45).any():  # example threshold
            suggestions.append("Some posture parameters are high, consider ergonomic adjustments.")

    return suggestions
