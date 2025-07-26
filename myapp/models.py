from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class ESP32Data(models.Model):
    param1 = models.FloatField(null=True, blank=True)
    param2 = models.FloatField(null=True, blank=True)
    param3 = models.FloatField(null=True, blank=True)

    # Add quaternion fields
    q_w = models.FloatField(null=True, blank=True)
    q_x = models.FloatField(null=True, blank=True)
    q_y = models.FloatField(null=True, blank=True)
    q_z = models.FloatField(null=True, blank=True)

    timestamp = models.DateTimeField(auto_now_add=True)
    prediction = models.CharField(max_length=255, null=True, blank=True)
    predicted = models.BooleanField(default=False, null=True, blank=True)

    def __str__(self):
        return f"Data at {self.timestamp}"

from django.db import models

class AgentTunnel(models.Model):
    user_id = models.CharField(max_length=100, unique=True)
    tunnel_url = models.URLField()
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user_id} -> {self.tunnel_url}"
    


class SensorReading(models.Model):
    device_id = models.CharField(max_length=100)  # ESP device identifier
    temperature = models.FloatField()
    humidity = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.device_id} - {self.timestamp}"
    

class GPSReading(models.Model):
    latitude = models.FloatField()
    longitude = models.FloatField()
    gps_time = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)