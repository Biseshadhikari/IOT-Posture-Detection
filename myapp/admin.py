from django.contrib import admin

# Register your models here.
from .models import *
admin.site.register(ESP32Data)


from .models import AgentTunnel

admin.site.register(AgentTunnel)