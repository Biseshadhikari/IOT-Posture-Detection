# app/management/commands/analyze_posture.py
from django.core.management.base import BaseCommand
from django.conf import settings
from pathlib import Path
import pandas as pd
from myapp.models import ESP32Data  # adjust import to your app
from myapp.services.posture_features import queryset_to_df
from myapp.services.posture_ml import PostureModels

class Command(BaseCommand):
    help = "Train posture ML models (RF, KMeans, IsolationForest) from ESP32Data without schema changes."

    def handle(self, *args, **options):
        qs = ESP32Data.objects.all().order_by("timestamp")
        df = queryset_to_df(qs)
        if df.empty:
            self.stdout.write(self.style.WARNING("No data to train."))
            return
        models_dir = Path(getattr(settings, "ML_MODELS_DIR", Path(settings.BASE_DIR) / "ml_models"))
        pm = PostureModels(models_dir)
        report = pm.fit_all(df)
        self.stdout.write(self.style.SUCCESS(f"Training complete: {report}"))
