from django.db import models
from django.utils import timezone

class ConversationSession(models.Model):
    """Store conversation sessions for users."""
    session_id = models.CharField(max_length=255, unique=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user_agent = models.CharField(max_length=500, blank=True, null=True)
    
    class Meta:
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"Session {self.session_id[:8]}... ({self.created_at})"


class Message(models.Model):
    """Store individual messages in a conversation."""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]
    
    session = models.ForeignKey(ConversationSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    tokens_used = models.IntegerField(default=0)  # Track token usage for monitoring
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']
        indexes = [
            models.Index(fields=['session', '-created_at']),
        ]
    
    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."
