import json
from django.http import JsonResponse
from django.shortcuts import render
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

# Web login view
def web_login(request):
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
    UserToken.objects.filter(user=user).delete()
    logout(request)  # Django logout
    return redirect('login')


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

    device, _ = ESPDevice.objects.get_or_create(device_id=device_id)

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
    new_data.save()

    # Prediction
    feature_data = pd.DataFrame([[ax, ay, az]], columns=['accx','accy','accz'])
    y_pred = rf_classifier.predict(feature_data)
    prediction = label_encoder.inverse_transform(y_pred)[0]
    alert = (prediction == 'bad')

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







from django.http import JsonResponse
from django.utils import timezone
from datetime import timedelta
import random

from .models import ESP32Data

def generate_synthetic_data(request):
    data_to_create = []
    now = timezone.now()

    for i in range(2000):
        is_good = random.random() > 0.5

        if is_good:
            ax = random.uniform(-1.5, 1.5)
            ay = random.uniform(-1.5, 1.5)
            az = random.uniform(8.5, 10.2)
            prediction = 'good'
        else:
            ax = random.uniform(-6.0, 6.0)
            ay = random.uniform(-6.0, 6.0)
            az = random.uniform(3.0, 8.0)
            prediction = 'bad'

        timestamp = now - timedelta(seconds=(2000 - i))

        data_to_create.append(
            ESP32Data(
                param1=ax,
                param2=ay,
                param3=az,
                prediction=prediction,
                user = request.user,
                predicted=False,
                timestamp=timestamp
            )
        )

    ESP32Data.objects.bulk_create(data_to_create)

    return JsonResponse({'message': '✅ 2000 synthetic data rows added successfully.'})


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from .models import ESP32Data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Posture naming based on centroid
def name_cluster(centroid):
    x, y, z = centroid
    ideal_z = 9.8
    # Adjusted thresholds for your data range
    if abs(x) < 0.5 and abs(y) < 0.7 and abs(z - ideal_z) < 0.8:
        return "Good Posture"
    elif abs(x) < 1.0 and abs(y) < 1.0 and abs(z - ideal_z) < 1.5:
        return "Slightly Off Posture"
    else:
        return "Poor Posture"

def recommend_posture(centroid):
    x, y, z = centroid
    ideal_z = 9.8
    if abs(x) < 0.5 and abs(y) < 0.7 and abs(z - ideal_z) < 0.8:
        return "Good Posture - Keep it up! Your posture is well balanced."
    elif abs(x) < 1.0 and abs(y) < 1.0 and abs(z - ideal_z) < 1.5:
        return ("Slightly Off Posture - Try to adjust your position a bit "
                "to avoid discomfort.")
    else:
        return ("Poor Posture - Please correct your posture soon to prevent pain "
                "or injury.")

@csrf_exempt
def posture_clusters(request):
    # Get all data for the user
    data_qs = ESP32Data.objects.filter(user=request.user)
    df = pd.DataFrame(list(data_qs.values('param1', 'param2', 'param3')))
    
    if df.empty:
        return JsonResponse({'error': 'No data available'}, status=404)
    
    df.rename(columns={'param1': 'x', 'param2': 'y', 'param3': 'z'}, inplace=True)
    
    # Remove extreme outliers (realistic human posture ranges)
    df = df[(df['x'].abs() < 2) & (df['y'].abs() < 2) & (df['z'].between(8, 12))]
    
    if df.empty:
        return JsonResponse({'error': 'No valid posture data after filtering outliers'}, status=404)
    
    # Scale features before clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['x', 'y', 'z']])
    
    # KMeans clustering
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    df['cluster'] = labels
    
    cluster_summary = []
    for i in range(n_clusters):
        cluster_points = df[df['cluster'] == i]
        centroid_scaled = kmeans.cluster_centers_[i]
        centroid_original = scaler.inverse_transform([centroid_scaled])[0].tolist()
        count = len(cluster_points)
        name = name_cluster(centroid_original)
        recommendation = recommend_posture(centroid_original)
        
        cluster_summary.append({
            'cluster': i,
            'name': name,
            'centroid': centroid_original,
            'count': count,
            'recommendation': recommendation,
        })
    
    points = df[['x', 'y', 'z', 'cluster']].to_dict(orient='records')

    return JsonResponse({
        'clusters': cluster_summary,
        'points': points
    })


# Optional: view to render a template
def posture_clusters_view(request):
    return render(request, 'posture_clusters.html')




def live_chart_view(request):
    return render(request, 'live_chart.html')


def get_latest_data(request):
    try:
        latest = ESP32Data.objects.latest('timestamp')
        return JsonResponse({
            "timestamp": latest.timestamp.strftime("%H:%M:%S"),
            "param1": latest.param1,
            "param2": latest.param2,
            "param3": latest.param3,
        })
    except ESP32Data.DoesNotExist:
        return JsonResponse({
            "timestamp": "",
            "param1": None,
            "param2": None,
            "param3": None,
        })
    

def mpu_3d_view(request):
    # Renders the template with 3D visualization
    return render(request, '3d.html')

def get_latest_orientation(request):
    latest = ESP32Data.objects.order_by('-timestamp').first()
    return JsonResponse({
        'param1': latest.param1 if latest else 0,
        'param2': latest.param2 if latest else 0,
        'param3': latest.param3 if latest else 0,
    })




from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse
from .models import UserToken, ESPDevice
import uuid
