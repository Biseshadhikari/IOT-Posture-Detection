from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from core.models import TunnelURL

class Command(BaseCommand):
    help = 'Cleanup stale tunnel URLs'
    
    def handle(self, *args, **options):
        # Mark tunnels as inactive if not updated in 5 minutes
        cutoff_time = timezone.now() - timedelta(minutes=5)
        
        stale_tunnels = TunnelURL.objects.filter(
            is_active=True,
            last_updated__lt=cutoff_time
        )
        
        count = stale_tunnels.count()
        stale_tunnels.update(is_active=False)
        
        self.stdout.write(
            self.style.SUCCESS(f'Marked {count} stale tunnels as inactive')
        )