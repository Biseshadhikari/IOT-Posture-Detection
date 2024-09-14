import json
from django.http import JsonResponse
from django.shortcuts import render
from .models import ESP32Data

def receive_data(request):
    if request.method == 'POST':
        try:
            # Parse the received JSON data
            data = json.loads(request.body)
            param1 = data.get('param1')
            param2 = data.get('param2')
            param3 = data.get('param3')
            param4 = data.get('param4')
            param5 = data.get('param5')

            # Save the received data into the database
            esp32_data = ESP32Data(
                param1=param1, param2=param2, param3=param3, param4=param4, param5=param5
            )
            esp32_data.save()

            # Return a success message
            return JsonResponse({'message': 'Data saved successfully.'}, status=201)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'message': 'Send data via POST method'}, status=200)



def predict_view(request):
    # Retrieve data from the database
    latest_data = ESP32Data.objects.filter(predicted=False).order_by('-timestamp')
    if latest_data:
        # Make predictions using the latest data
        # For demonstration purposes, we'll use a simple linear regression model

        single_data = latest_data[0] # latest sensor data received
        param1 = single_data.param1
        param2 = single_data.param2
        param3 = single_data.param3
        param4 = single_data.param4
        param5 = single_data.param5
        # Make predictions here 
        # USE MODEL HERE


        # Let assume this is predection result
        prediction = None

        # Save the prediction to the database
        single_data.prediction = None
        single_data.predicted = True
        single_data.save()
    
        
        # Delete all the stale data or updated it to perdicted value
        # ESP32Data.objects.filter(predicted=False).update(predicted=True)

        return render(request,'predictions.html',{
            'param1': param1,
            'param2': param2,
            'param3': param3,
            'param4': param4,
            'param5': param5,
            'prediction':prediction
            
        })

    else:
        return render(request,'predictions.html',{
            "error":"No data to predict"
            
        })

        

