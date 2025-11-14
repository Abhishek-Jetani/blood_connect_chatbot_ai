"""
In-memory conversation storage (no database).
For production, replace with Redis, file-based storage, or a document database.
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Optional


class ConversationSession:
    """In-memory conversation session storage."""
    
    _sessions: Dict[str, 'ConversationSession'] = {}
    
    def __init__(self, session_id: str, user_agent: str = ""):
        self.session_id = session_id
        self.user_agent = user_agent
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.messages: List['Message'] = []
        ConversationSession._sessions[session_id] = self
    
    @classmethod
    def get_or_create(cls, session_id: str, user_agent: str = "") -> tuple:
        """Get existing session or create new one. Returns (session, created)."""
        if session_id in cls._sessions:
            return cls._sessions[session_id], False
        session = cls(session_id, user_agent)
        return session, True
    
    @classmethod
    def get(cls, session_id: str) -> Optional['ConversationSession']:
        """Retrieve a session by ID."""
        return cls._sessions.get(session_id)
    
    def __str__(self):
        return f"Session {self.session_id[:8]}... ({self.created_at})"


class Message:
    """In-memory message storage within a session."""
    
    ROLE_CHOICES = ('user', 'assistant')
    
    def __init__(self, session: ConversationSession, role: str, content: str, tokens_used: int = 0):
        if role not in self.ROLE_CHOICES:
            raise ValueError(f"Invalid role: {role}")
        self.session = session
        self.role = role
        self.content = content
        self.tokens_used = tokens_used
        self.created_at = datetime.now()
        session.messages.append(self)
    
    @classmethod
    def get_by_session(cls, session: ConversationSession) -> List['Message']:
        """Retrieve all messages for a session."""
        return session.messages
    
    def to_dict(self):
        """Convert message to dictionary for JSON serialization."""
        return {
            'role': self.role,
            'content': self.content,
            'created_at': self.created_at.isoformat(),
            'tokens_used': self.tokens_used,
        }
    
    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."
