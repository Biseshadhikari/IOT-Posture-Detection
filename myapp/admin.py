from django.contrib import admin

# Register your models here.
from .models import *
admin.site.register(ESP32Data)



admin.site.register(UserToken)
admin.site.register(ESPDevice)