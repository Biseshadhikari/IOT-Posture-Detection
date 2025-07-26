from django.urls import path
from . import views

urlpatterns = [
    path('receive/', views.receive_data, name='receive_data'),
    path('predict/', views.predict_view, name='predict'),
    path('generate-synthetic-data/', views.generate_synthetic_data, name='generate-synthetic-data'),
    path('posture/clusters/', views.posture_clusters, name='posture_clusters'),
    path('posture/clusters/view/', views.posture_clusters_view, name='posture_clusters_view'),
    path('live-chart/', views.live_chart_view, name='live_chart'),
    path('latest_data/', views.get_latest_data, name='latest_data'),
    path('3d/', views.mpu_3d_view, name='3d'),
    path('latest-quaternion/', views.latest_quaternion, name='latest_quaternion'),
    path('both/', views.both, name='live_chart'),
    path('receive-temp-humidity/', views.temp_humidity_receiver, name='receive_temp_humidity'),
    path('dht-data/', views.dht_data, name='latest_dht_data'),
    path('sensor/', views.temperature_humidity_view, name='temperature_humidity'),
        path('api/latest-gps/', views.latest_gps_data, name='latest_gps_data'),
            path('map/', views.gps_map_view, name='gps_map'),








 



]