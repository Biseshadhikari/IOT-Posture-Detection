import json
from django.http import JsonResponse
from django.shortcuts import render,redirect
from .models import ESP32Data
import joblib
import pandas as pd
from django.contrib.auth.decorators import login_required

from django.views.decorators.csrf import csrf_exempt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


from django.http import JsonResponse
import json
import pandas as pd
import joblib

import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.sessions.models import Session
from django.utils import timezone
from .models import ESP32Data, UserToken, ESPDevice
import pandas as pd
import joblib
from django.shortcuts import render
from django.utils import timezone
from .models import ESP32Data
from datetime import timedelta
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from django.contrib.auth import login,logout,authenticate

# Web login view
def web_login(request):
    if request.user.is_authenticated: 
        return redirect('/')
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)  # Django session login

            # Generate or update token for ESP
            token = UserToken.generate_token(user)

            return redirect('/')  # Redirect to your dashboard or homepage
        else:
            return render(request, 'login.html', {'error': 'Invalid credentials'})
    return render(request, 'login.html')


# Web logout view

def web_logout(request):
    user = request.user
    # Delete the token when user logs out
    logout(request)  # Django logout
    return redirect('/login')


# Load model and label encoder
rf_classifier = joblib.load('random_forest_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')


# ---------------------------
# ESP Polling for token
# ---------------------------

@csrf_exempt
def esp_poll(request):
    device_id = request.GET.get('device_id')
    if not device_id:
        return JsonResponse({'status': 'error', 'message': 'No device_id provided'}, status=400)

    device= ESPDevice.objects.all().first()
    device.device_id = device_id
    device.save()

    if device.user:
        token_str = UserToken.generate_token(device.user)
        return JsonResponse({'status': 'logged_in', 'token': token_str})
    else:
        return JsonResponse({'status': 'logged_out', 'token': ''})


@csrf_exempt
def receive_data(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)

    # Check token
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    print(token)
    if not token:
        return JsonResponse({'error': 'No token provided'}, status=401)


    token_obj = UserToken.objects.get(token=token)
    print(token_obj)

    user = token_obj.user
    
    

    # Check device
    device_id = request.headers.get('Device-ID')
    try:
        device = ESPDevice.objects.get(device_id=device_id)
        if device.user != user:
            return JsonResponse({'error': 'Device not assigned to user'}, status=403)
    except ESPDevice.DoesNotExist:
        return JsonResponse({'error': 'Device not registered'}, status=403)

    # Process sensor data
    data = json.loads(request.body.decode('utf-8'))
    ax = float(data.get('ax', None))
    ay = float(data.get('ay', None))
    az = float(data.get('az', None))

    if ax is None or ay is None or az is None:
        return JsonResponse({'error': 'Missing accelerometer data'}, status=400)

    # Save with user
    new_data = ESP32Data(
        user=user,               # assign the user
        param1=ax,
        param2=ay,
        param3=az,
        predicted=False,
        # prediction=prediction   # <-- save prediction here

    )
    

    # Prediction
    feature_data = pd.DataFrame([[ax, ay, az]], columns=['accx','accy','accz'])
    y_pred = rf_classifier.predict(feature_data)
    prediction = label_encoder.inverse_transform(y_pred)[0]
    alert = (prediction == 'bad')
    new_data.prediction = prediction
    new_data.save()

    return JsonResponse({'prediction': prediction, 'alert': alert})









# Function to handle predictions using the stored data and pre-trained model
@login_required
def predict_view(request):
    # Retrieve the latest non-predicted data from the database
    latest_data = ESP32Data.objects.filter(predicted=False,user = request.user).order_by('-timestamp')

    if latest_data.exists():
        # Get the latest sensor data received
        single_data = latest_data.first()
        param1 = single_data.param1
        param2 = single_data.param2
        param3 = single_data.param3
     
        try:
            # Load the pre-trained RandomForest model and LabelEncoder
            try:
                loaded_rf_classifier = joblib.load('random_forest_model.pkl')
                loaded_label_encoder = joblib.load('label_encoder.pkl')
            except FileNotFoundError:
                return render(request, 'predictions.html', {
                    "error": "Model files not found. Make sure the models are properly saved."
                })
            except Exception as e:
                return render(request, 'predictions.html', {
                    "error": f"Error loading model: {str(e)}"
                })

            # Prepare the data for prediction (list of lists)
            feature_data = [[param1, param2, param3]]

            # Make predictions using the loaded model
            y_pred = loaded_rf_classifier.predict(feature_data)

            # Decode the prediction using LabelEncoder
            prediction = loaded_label_encoder.inverse_transform(y_pred)[0]

            # Update the prediction in the database and mark it as predicted
            single_data.prediction = prediction
            single_data.predicted = True
            single_data.save()
            # // deleting stale data
            
            latest_data.exclude(id=single_data.id).delete()
            # Render the prediction in the template
            return render(request, 'predictions.html', {
                'param1': param1,
                'param2': param2,
                'param3': param3,
                'prediction': prediction
            })

        except Exception as e:
            return render(request, 'predictions.html', {
                "error": f"Prediction failed: {str(e)}"
            })

    else:
        # No data available for prediction
        return render(request, 'predictions.html', {
            "error": "No data to predict"
        })


from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from myapp.models import ESP32Data
from myapp.services.posture_features import (
    queryset_to_df,
    daily_summary,
    weekly_summary,
    peak_bad_hours,
    longest_good_streak,
    suggestions_from_analytics
)
from .services.posture_ml import PostureModels

@login_required
def posture_analysis(request):
    user = request.user

    # Fetch user's data
    qs = ESP32Data.objects.filter(user=user).order_by("timestamp")
    df = queryset_to_df(qs)

    # Load ML models from fixed path (update this path to your system)
    models_dir = Path("/Users/biseshadhikari/finalproj/Posture-Detection-App/ml_models")
    pm = PostureModels(models_dir)

    # Predict posture using ML if data exists
    try:
        pred_df = pm.predict_all(df) if not df.empty else df
    except Exception as e:
        # Log exception (optional)
        print(f"ML prediction failed: {e}")
        pred_df = df

    # --- Analytics ---
    daily = daily_summary(pred_df)
    weekly = weekly_summary(pred_df)
    peaks = peak_bad_hours(pred_df)
    streak = longest_good_streak(pred_df)

    # --- Cluster stats (if clustering is applied) ---
    cluster_stats = []
    if not pred_df.empty and "cluster" in pred_df.columns:
        cluster_counts = pred_df["cluster"].value_counts(normalize=True)
        cluster_stats = [
            {"cluster": int(idx), "share": float(share)}
            for idx, share in cluster_counts.items()
        ]

    # --- Recent anomalies ---
    anomalies = []
    if not pred_df.empty and "is_anomaly" in pred_df.columns:
        recent_anomalies = pred_df[pred_df["is_anomaly"] == 1].tail(20)
        anomalies = [
            {
                "timestamp": str(r.ts),
                "param1": float(r.param1),
                "param2": float(r.param2),
                "param3": float(r.param3)
            }
            for r in recent_anomalies.itertuples(index=False)
        ]

    # --- Suggestions (rule-based or from ML) ---
    suggestions = suggestions_from_analytics(daily, pred_df, peaks, streak)

    # --- Prepare JSON response ---
    return JsonResponse({
        "daily": daily.tail(60).to_dict(orient="records") if not daily.empty else [],
        "weekly": weekly.tail(26).to_dict(orient="records") if not weekly.empty else [],
        "peak_bad_hours": peaks,
        "longest_good_streak_days": streak,
        "cluster_shares": cluster_stats,
        "recent_anomalies": anomalies,
        "suggestions": suggestions,
    }, safe=False)


@login_required
def posture_dashboard(request):
    """
    Render the posture analysis dashboard (frontend visualization).
    Uses the existing posture_analysis API for data.
    """
    return render(request, "posture_dashboard.html")    