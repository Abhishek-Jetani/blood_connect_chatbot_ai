from django.contrib import admin
from .models import ConversationSession, Message


@admin.register(ConversationSession)
class ConversationSessionAdmin(admin.ModelAdmin):
    list_display = ('session_id_short', 'message_count', 'created_at', 'updated_at')
    list_filter = ('created_at', 'updated_at')
    search_fields = ('session_id',)
    readonly_fields = ('session_id', 'created_at', 'updated_at')
    
    def session_id_short(self, obj):
        return f"{obj.session_id[:8]}..."
    session_id_short.short_description = 'Session ID'
    
    def message_count(self, obj):
        return obj.messages.count()
    message_count.short_description = 'Messages'
    
    def has_add_permission(self, request):
        return False


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'session_short', 'role', 'content_preview', 'created_at')
    list_filter = ('role', 'created_at', 'session__session_id')
    search_fields = ('content', 'session__session_id')
    readonly_fields = ('session', 'role', 'content', 'created_at')
    
    def session_short(self, obj):
        return f"{obj.session.session_id[:8]}..."
    session_short.short_description = 'Session'
    
    def content_preview(self, obj):
        return obj.content[:50] + "..." if len(obj.content) > 50 else obj.content
    content_preview.short_description = 'Content'
    
    def has_add_permission(self, request):
        return False
