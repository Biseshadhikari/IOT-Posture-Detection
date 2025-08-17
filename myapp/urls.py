from django.urls import path
from . import views

urlpatterns = [

    path('receive/', views.receive_data, name='receive_data'),
    path('predict/', views.predict_view, name='predict'),
   

    path('esp_poll/', views.esp_poll, name='esp_poll'),
     path("api/posture/analysis/", views.posture_analysis, name="posture_analysis"),

    path('login/', views.web_login, name='login'),
    path('logout/', views.web_logout, name='logout'),
        path("posture/dashboard/", views.posture_dashboard, name="posture_dashboard"),





 



]