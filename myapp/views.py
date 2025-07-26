import json
from django.http import JsonResponse
from django.shortcuts import render
from .models import ESP32Data
import joblib
import pandas as pd

from django.views.decorators.csrf import csrf_exempt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


from django.http import JsonResponse
import json
import pandas as pd
import joblib

import json
import pandas as pd
import joblib
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import *

# Load model and label encoder once at module load time
rf_classifier = joblib.load('random_forest_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import ESP32Data, SensorReading
@csrf_exempt
def receive_data(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))

        # MPU6050 / posture data
        ax = float(data.get('ax'))
        ay = float(data.get('ay'))
        az = float(data.get('az'))

        q_w = float(data.get('q_w'))
        q_x = float(data.get('q_x'))
        q_y = float(data.get('q_y'))
        q_z = float(data.get('q_z'))

        # Device ID (optional, not used in GPSReading model)
        device_id = data.get('device_id', 'unknown-device')

        # Temperature & Humidity (optional)
        temp = data.get('temperature')
        hum = data.get('humidity')

        # GPS data
        latitude = data.get('gps_lat')
        longitude = data.get('gps_lng')
        gps_time = data.get('gps_time', '')  # Could be a string timestamp, default empty

        # Save posture data
        new_posture = ESP32Data(
            param1=ax,
            param2=ay,
            param3=az,
            q_w=q_w,
            q_x=q_x,
            q_y=q_y,
            q_z=q_z,
            predicted=False
        )
        new_posture.save()

        # Save temperature and humidity
        if temp is not None and hum is not None:
            try:
                temp_val = float(temp)
                hum_val = float(hum)
                SensorReading.objects.create(
                    device_id=device_id,
                    temperature=temp_val,
                    humidity=hum_val
                )
            except ValueError:
                pass

        # Save GPS data if latitude and longitude are present
        if latitude is not None and longitude is not None:
            try:
                lat_val = float(latitude)
                lon_val = float(longitude)
                gps_time_val = str(gps_time) if gps_time else ""

                GPSReading.objects.create(
                    latitude=lat_val,
                    longitude=lon_val,
                    gps_time=gps_time_val
                )
            except ValueError:
                pass  # Ignore bad data

        prediction = 'good'
        alert = False

        return JsonResponse({'prediction': prediction, 'alert': alert})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)




def latest_gps_data(request):
    latest = GPSReading.objects.order_by('-created_at').first()
    if latest:
        data = {
            'latitude': latest.latitude,
            'longitude': latest.longitude,
            'timestamp': latest.created_at.isoformat(),
        }
    else:
        data = {
            'latitude': 27.681,  # default lat
            'longitude': 85.32,  # default lng
            'timestamp': None,
        }
    return JsonResponse(data)

def gps_map_view(request):
  
    lat, lng = 27.681, 85.32  # Default location (Biratnagar, Nepal)

    context = {
        'lat': lat,
        'lng': lng
    }
    return render(request, 'gps_map.html', context)








# Function to handle predictions using the stored data and pre-trained model
def predict_view(request):
    # Retrieve the latest non-predicted data from the database
    latest_data = ESP32Data.objects.filter(predicted=False).order_by('-timestamp')

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
                predicted=True,
                timestamp=timestamp
            )
        )

    ESP32Data.objects.bulk_create(data_to_create)

    return JsonResponse({'message': '✅ 2000 synthetic data rows added successfully.'})


import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import ESP32Data
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def recommend_posture(centroid):
    x, y, z = centroid
    # Define thresholds based on normalized scale
    # Post scaling, values ~0, so use distance from ideal (0,0,gravity_norm)
    # gravity norm after scaling is ~ mean of scaled Z

    # For simplicity, use original scale with a margin

    ideal_z = 9.8
    x_abs = abs(x)
    y_abs = abs(y)
    z_diff = abs(z - ideal_z)

    if x_abs < 0.15 and y_abs < 0.15 and z_diff < 0.5:
        return "Good Posture - Keep it up! Your posture is well balanced."
    elif x_abs < 0.4 and y_abs < 0.4 and z_diff < 1.0:
        return ("Slightly Off Posture - Try to adjust your position a bit "
                "to avoid discomfort.")
    else:
        return ("Poor Posture - Please correct your posture soon to prevent pain "
                "or injury.")

def name_cluster(centroid):
    x, y, z = centroid
    ideal_z = 9.8
    if abs(x) < 0.15 and abs(y) < 0.15 and abs(z - ideal_z) < 0.5:
        return "Good Posture"
    elif abs(x) < 0.4 and abs(y) < 0.4 and abs(z - ideal_z) < 1.0:
        return "Slightly Off Posture"
    else:
        return "Poor Posture"

@csrf_exempt
def posture_clusters(request):
    data_qs = ESP32Data.objects.all()
    df = pd.DataFrame(list(data_qs.values('param1', 'param2', 'param3')))
    if df.empty:
        return JsonResponse({'error': 'No data available'}, status=404)
    
    df.rename(columns={'param1': 'x', 'param2': 'y', 'param3': 'z'}, inplace=True)
    
    # Scale features before clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['x', 'y', 'z']])
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    df['cluster'] = labels
    
    cluster_summary = []
    for i in range(3):
        cluster_points = df[df['cluster'] == i]
        
        # To interpret centroid in original scale, inverse transform
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

def posture_clusters_view(request):
    return render(request, 'posture_clusters.html')




def live_chart_view(request):
    return render(request, 'live_chart.html')

from django.http import JsonResponse
from math import sqrt
from .models import ESP32Data

def get_latest_data(request):
    try:
        latest = ESP32Data.objects.latest('timestamp')

        ax = latest.param1
        ay = latest.param2
        az = latest.param3

        total_acceleration = sqrt(ax**2 + ay**2 + az**2)

        return JsonResponse({
            "timestamp": latest.timestamp.strftime("%H:%M:%S"),
            "param1": ax,
            "param2": ay,
            "param3": az,
            "acceleration": round(total_acceleration, 3),
        })
    except ESP32Data.DoesNotExist:
        return JsonResponse({
            "timestamp": "",
            "param1": None,
            "param2": None,
            "param3": None,
            "acceleration": None,
        })

    

def mpu_3d_view(request):
    # Renders the template with 3D visualization
    return render(request, '3d.html')

def latest_quaternion(request):
    # Get latest data point
    latest = ESP32Data.objects.order_by('-timestamp').first()
    if not latest:
        return JsonResponse({'error': 'No data'}, status=404)

    data = {
        'q_w': latest.q_w,
        'q_x': latest.q_x,
        'q_y': latest.q_y,
        'q_z': latest.q_z,
        'timestamp': latest.timestamp.isoformat()
    }
    return JsonResponse(data)




def both(request):
    return render(request,'both.html')

import json
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from .models import SensorReading

@csrf_exempt
def temp_humidity_receiver(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            device_id = data.get("device_id")
            temperature = float(data.get("temperature"))
            humidity = float(data.get("humidity"))

            if not device_id:
                return HttpResponseBadRequest("Missing device_id")

            SensorReading.objects.create(
                device_id=device_id,
                temperature=temperature,
                humidity=humidity
            )
            return JsonResponse({"message": "Data saved successfully"})
        except Exception as e:
            return HttpResponseBadRequest(f"Invalid data: {e}")

    return HttpResponseBadRequest("Only POST allowed")

def dht_data(request):
    readings = SensorReading.objects.order_by('-timestamp')[:20]
    data = []
    for r in readings:
        data.append({
            "device_id": r.device_id,
            "temperature": r.temperature,
            "humidity": r.humidity,
            "timestamp": r.timestamp.isoformat()
        })
    return JsonResponse(data, safe=False)



def temperature_humidity_view(request):
    return render(request, 'temp.html')