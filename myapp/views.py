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
from .models import ESP32Data

# Load model and label encoder once at module load time
rf_classifier = joblib.load('random_forest_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@csrf_exempt
def receive_data(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)
    
    try:
        data = json.loads(request.body.decode('utf-8'))

        # Validate and convert inputs
        ax = float(data.get('ax', None))
        ay = float(data.get('ay', None))
        az = float(data.get('az', None))

        if ax is None or ay is None or az is None:
            return JsonResponse({'error': 'Missing accelerometer data'}, status=400)

        # Save data to DB
        new_data = ESP32Data(param1=ax, param2=ay, param3=az, predicted=False)
        new_data.save()

        # Prepare feature for prediction
        feature_data = pd.DataFrame([[ax, ay, az]], columns=['accx', 'accy', 'accz'])

        # Predict
        y_pred = rf_classifier.predict(feature_data)
        prediction = label_encoder.inverse_transform(y_pred)[0]

        # Define alert condition
        alert = (prediction == 'bad')
        print(f'Prediction: {prediction}, Alert: {alert}')

        return JsonResponse({'prediction': prediction, 'alert': alert})

    except (ValueError, TypeError) as e:
        return JsonResponse({'error': f'Invalid input data: {str(e)}'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)










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
