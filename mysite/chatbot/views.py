import json
import os
import uuid
import logging
from pathlib import Path
from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from .models import ConversationSession, Message
from .blood_assistant import answer_question, get_assistant

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - [DJANGO] %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def chat_ui(request):
    """Render the chat UI template."""
    logger.info("Chat UI requested")
    return render(request, 'chatbot/chat.html')


# Blood Assistant Integration
# Uses free, open-source models: DistilBERT, FLAN-T5, Sentence-Transformers
# No API keys required - everything runs locally!

logger.info("Initializing Blood Assistant on first import...")
try:
    assistant = get_assistant()
    logger.info("✓ Blood Assistant initialized successfully")
except Exception as e:
    logger.error(f"✗ Failed to initialize Blood Assistant: {e}")
    assistant = None


@csrf_exempt
def send_message(request):
    """Endpoint that handles chat messages with debugging"""
    logger.info("=" * 70)
    logger.info("SEND_MESSAGE ENDPOINT CALLED")
    logger.info("=" * 70)
    
    if request.method != 'POST':
        logger.warning(f"Invalid method: {request.method}, expected POST")
        return HttpResponseBadRequest('POST required')

    try:
        payload = json.loads(request.body.decode('utf-8'))
        logger.debug(f"Payload received: {json.dumps(payload, indent=2)}")
    except Exception as e:
        logger.error(f"JSON parsing error: {e}")
        return HttpResponseBadRequest('Invalid JSON')
    
    text = payload.get('text', '').strip()
    session_id = payload.get('session_id', '')

    if not text:
        logger.warning("No text provided in request")
        return HttpResponseBadRequest('text required')

    logger.info(f"Processing message: '{text[:100]}'")
    logger.info(f"Session ID: {session_id if session_id else 'NEW'}")

    # Create or get session
    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"Created new session: {session_id}")
    
    # Get or create session (in-memory)
    session, created = ConversationSession.get_or_create(
        session_id=session_id,
        user_agent=request.META.get('HTTP_USER_AGENT', '')
    )
    logger.debug(f"Session status: {'CREATED' if created else 'EXISTING'}")
    
    # Store user message (in-memory)
    user_msg = Message(session=session, role='user', content=text)
    logger.debug(f"User message stored: {user_msg}")
    
    # Get conversation history for context (exclude current message)
    history = [{'role': m.role, 'content': m.content} for m in session.messages[:-1]]
    logger.debug(f"Conversation history: {len(history)} previous messages")

    # Check if assistant is initialized
    if not assistant:
        logger.error("Blood Assistant not initialized!")
        return JsonResponse({
            'ok': False,
            'error': 'AI assistant not initialized. Check server logs for details.',
            'session_id': session_id
        }, status=503)

    try:
        logger.info("Calling Blood Assistant to generate answer...")
        
        # Use Blood Assistant with full debugging
        result = answer_question(
            question=text,
            conversation_history=history,
            debug=True
        )
        
        answer = result['answer']
        intent = result.get('intent', 'unknown')
        confidence = result.get('confidence', 0.0)
        sources = result.get('sources', [])
        
        logger.info(f"Answer generated successfully")
        logger.info(f"  Intent: {intent}")
        logger.info(f"  Confidence: {confidence:.2f}")
        logger.info(f"  Sources used: {len(sources)}")
        logger.debug(f"  Answer: {answer[:100]}...")
        
        # Store assistant response (in-memory)
        assistant_msg = Message(session=session, role='assistant', content=answer)
        logger.debug(f"Assistant message stored: {assistant_msg}")
        
        return JsonResponse({
            'ok': True,
            'reply': {
                'type': 'text',
                'text': answer,
                'intent': intent,
                'confidence': confidence,
                'sources': sources
            },
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"Answer generation failed: {e}", exc_info=True)
        return JsonResponse({
            'ok': False,
            'error': f'Answer generation failed: {str(e)}',
            'session_id': session_id
        }, status=500)


@csrf_exempt
def get_history(request):
    """Retrieve conversation history for a session."""
    logger.debug("GET_HISTORY endpoint called")
    
    if request.method != 'POST':
        logger.warning(f"Invalid method: {request.method}, expected POST")
        return HttpResponseBadRequest('POST required')

    try:
        payload = json.loads(request.body.decode('utf-8'))
    except Exception as e:
        logger.error(f"JSON parsing error: {e}")
        return HttpResponseBadRequest('Invalid JSON')
    
    session_id = payload.get('session_id', '')
    
    if not session_id:
        logger.warning("No session_id provided")
        return JsonResponse({'ok': False, 'error': 'session_id required'}, status=400)
    
    logger.debug(f"Retrieving history for session: {session_id}")
    
    # Retrieve session from in-memory store
    session = ConversationSession.get(session_id)
    if not session:
        logger.info(f"Session {session_id} does not exist")
        return JsonResponse({
            'ok': True,
            'session_id': session_id,
            'messages': [],
            'note': 'Session does not exist yet'
        })
    
    # Retrieve all messages in the session
    messages = [m.to_dict() for m in session.messages]
    logger.info(f"Retrieved {len(messages)} messages from session {session_id}")
    
    return JsonResponse({
        'ok': True,
        'session_id': session_id,
        'messages': messages,
        'created_at': session.created_at.isoformat(),
        'updated_at': session.updated_at.isoformat()
    })


@csrf_exempt
def upload_file(request):
    """Handle image/audio/file uploads and return an URL. This is a demo using local MEDIA_ROOT."""
    if request.method != 'POST':
        return HttpResponseBadRequest('POST required')

    if 'file' not in request.FILES:
        return HttpResponseBadRequest('file required')

    f = request.FILES['file']
    # Simple validation
    if f.size > 10 * 1024 * 1024:
        return HttpResponseBadRequest('file too large')

    upload_dir = Path(settings.MEDIA_ROOT) / 'uploads'
    os.makedirs(upload_dir, exist_ok=True)

    dest_path = upload_dir / f.name
    with open(dest_path, 'wb') as out:
        for chunk in f.chunks():
            out.write(chunk)

    url = settings.MEDIA_URL + f'uploads/{f.name}'
    return JsonResponse({'ok': True, 'url': url})
