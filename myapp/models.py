from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class ESP32Data(models.Model):
    user  = models.ForeignKey(User,on_delete=models.SET_NULL,null = True,blank=True)
    param1 = models.FloatField(null=True,blank=True)
    param2 = models.FloatField(null=True,blank=True)
    param3 = models.FloatField(null=True,blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    prediction = models.CharField(max_length=255,null=True,blank=True)
    predicted = models.BooleanField(default=False,null=True,blank=True)

    def __str__(self):
        return f"Data at {self.timestamp}"
