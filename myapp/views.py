import json
from django.http import JsonResponse
from django.shortcuts import render
from .models import ESP32Data
import joblib
from django.views.decorators.csrf import csrf_exempt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Function to handle data reception from ESP32
@csrf_exempt
def receive_data(request):
    if request.method == 'POST':
        try:
            # Parse the received JSON data
            data = json.loads(request.body)

            # Check if required params are present
            if 'ax' not in data or 'ay' not in data or 'az' not in data:
                return JsonResponse({'error': 'Missing parameters'}, status=400)

            param1 = data.get('ax')
            param2 = data.get('ay')
            param3 = data.get('az')

            # Validate data types
            try:
                param1 = float(param1)
                param2 = float(param2)
                param3 = float(param3)
            except ValueError:
                return JsonResponse({'error': 'Invalid data format, must be numbers'}, status=400)

            # Save the received data into the database
            esp32_data = ESP32Data(
                param1=param1, param2=param2, param3=param3
            )
            esp32_data.save()

            # Return a success message
            return JsonResponse({'message': 'Data saved successfully.'}, status=201)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON format'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'message': 'Send data via POST method'}, status=200)


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
