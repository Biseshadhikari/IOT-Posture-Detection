from django.urls import path
from . import views

urlpatterns = [
    path('receive/', views.receive_data, name='receive_data'),
    path('predict/', views.predict_view, name='predict'),
    path('generate-synthetic-data/', views.generate_synthetic_data, name='generate-synthetic-data'),
        path('posture/clusters/', views.posture_clusters, name='posture_clusters'),
            path('posture/clusters/view/', views.posture_clusters_view, name='posture_clusters_view'),




 



]