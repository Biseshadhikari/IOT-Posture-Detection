from django.contrib.auth.signals import user_logged_out
from django.dispatch import receiver
from .models import UserToken, ESPDevice

@receiver(user_logged_out)
def remove_token_and_device(sender, request, user, **kwargs):
    # Delete the user's token
    UserToken.objects.filter(user=user).delete()

    # Remove any device assignment
    ESPDevice.objects.filter(user=user).update(user=None)