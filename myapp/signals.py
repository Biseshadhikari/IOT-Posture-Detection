from django.contrib.auth.signals import user_logged_out
from django.dispatch import receiver
from .models import UserToken, ESPDevice

@receiver(user_logged_out)
def remove_token_and_device(sender, request, user, **kwargs):
    # Delete the user's token
    UserToken.objects.filter(user=user).delete()

    # Remove any device assignment
    ESPDevice.objects.filter(user=user).update(user=None)



from django.contrib.auth.signals import user_logged_in
from django.dispatch import receiver
from .models import ESPDevice

@receiver(user_logged_in)
def ensure_single_device(sender, request, user, **kwargs):


    try:
        device = ESPDevice.objects.all().first()
    except ESPDevice.DoesNotExist:
        return

    device.delete()
    ESPDevice.objects.create(user = request.user)