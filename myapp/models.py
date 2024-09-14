from django.db import models

# Create your models here.
from django.db import models

class ESP32Data(models.Model):
    param1 = models.FloatField(null=True,blank=True)
    param2 = models.FloatField(null=True,blank=True)
    param3 = models.FloatField(null=True,blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    prediction = models.CharField(max_length=255,null=True,blank=True)
    predicted = models.BooleanField(default=False,null=True,blank=True)

    def __str__(self):
        return f"Data at {self.timestamp}"
