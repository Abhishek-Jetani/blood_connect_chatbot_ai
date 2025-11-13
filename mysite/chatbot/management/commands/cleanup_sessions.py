from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from chatbot.models import ConversationSession, Message


class Command(BaseCommand):
    help = 'Clean up old conversation sessions (older than specified days)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days',
            type=int,
            default=30,
            help='Delete sessions older than this many days (default: 30)'
        )

    def handle(self, *args, **options):
        days = options['days']
        cutoff_date = timezone.now() - timedelta(days=days)
        
        # Find old sessions
        old_sessions = ConversationSession.objects.filter(updated_at__lt=cutoff_date)
        count = old_sessions.count()
        
        if count == 0:
            self.stdout.write(self.style.SUCCESS('No old sessions to delete.'))
            return
        
        self.stdout.write(f'Found {count} sessions older than {days} days.')
        
        # Delete messages first (due to foreign key)
        Message.objects.filter(session__in=old_sessions).delete()
        
        # Delete sessions
        old_sessions.delete()
        
        self.stdout.write(self.style.SUCCESS(
            f'Successfully deleted {count} old conversation sessions.'
        ))
