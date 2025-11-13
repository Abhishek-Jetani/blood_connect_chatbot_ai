import json
import os
import uuid
from pathlib import Path
from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from .models import ConversationSession, Message

# Try to use a local Hugging Face model (free) via transformers if available.
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    # Conversational pipeline and Conversation class may be available in newer transformers
    try:
        # import Conversation if available (older versions)
        from transformers import Conversation  # type: ignore
    except Exception:
        Conversation = None  # type: ignore
    _HAS_HF = True
except Exception:
    pipeline = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    Conversation = None  # type: ignore
    _HAS_HF = False


def chat_ui(request):
    """Render the chat UI template."""
    return render(request, 'chatbot/chat.html')


# Runtime LLM caller (OpenAI-compatible). Configure with OPENAI_API_KEY in environment.
# No external API calls here — prefer local Hugging Face models only (free).


def hf_answer(question, conversation_history=None):
    """Use a local Hugging Face seq2seq model to answer the question with conversation context.

    Set HUGGINGFACE_MODEL env var to choose model (default: google/flan-t5-small).
    Requires `transformers` and `torch` installed. Loads model on first call.
    
    Args:
        question: The user's current question
        conversation_history: List of previous {role, content} dicts for context
    """
    if not _HAS_HF:
        raise RuntimeError('Hugging Face transformers not available')

    model_name = os.environ.get('HUGGINGFACE_MODEL', 'google/flan-t5-small')
    # cache pipeline in module attribute
    if not hasattr(hf_answer, 'pipe') or hf_answer.pipe is None:
        # load model and tokenizer. Prefer a conversational pipeline if supported by the installed
        # transformers release and the chosen model. Fall back to text2text-generation.
        try:
            if Conversation is not None:
                hf_answer.pipe = pipeline('conversational', model=model_name, device=-1)
                hf_answer._is_conversational = True
            else:
                hf_answer.pipe = pipeline('text2text-generation', model=model_name, device=-1)
                hf_answer._is_conversational = False
        except Exception:
            # try manual model/tokenizer load for some environments
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                hf_answer.pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=-1)
                hf_answer._is_conversational = False
            except Exception as e2:
                raise RuntimeError(f'Failed to load Hugging Face model {model_name}: {e2}')

    # Build context from conversation history
    context = ""
    if conversation_history:
        # Include last 5 messages for context window
        recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        for msg in recent_history:
            role_label = "User" if msg['role'] == 'user' else "Assistant"
            context += f"{role_label}: {msg['content']}\n"
    
    prompt = (
        "You are a helpful medical chatbot specializing in blood donation information. "
        "Answer concisely and accurately about blood donation. If urgent, advise contacting medical professionals.\n\n"
        f"{context}"
        f"User: {question}\n"
        "Assistant:"
    )
    try:
        # Use the appropriate call depending on pipeline type
        if getattr(hf_answer, '_is_conversational', False):
            # conversational pipeline returns a Conversation object or a list; create one and extract
            try:
                conv = Conversation(prompt)
                res = hf_answer.pipe(conv)
                # Result may be a Conversation with .generated_responses
                if hasattr(conv, 'generated_responses') and conv.generated_responses:
                    text_out = conv.generated_responses[-1]
                else:
                    # Some versions return a list of Conversation objects
                    if isinstance(res, list) and res:
                        first = res[0]
                        text_out = getattr(first, 'generated_responses', [None])[-1] if first is not None else None
                    else:
                        text_out = str(res).strip()
            except Exception:
                # If conversational call fails, fall back to text generation
                out = hf_answer.pipe(prompt, max_new_tokens=256, do_sample=False)
                if isinstance(out, list) and out:
                    text_out = out[0].get('generated_text', str(out[0]))
                else:
                    text_out = str(out)
        else:
            out = hf_answer.pipe(prompt, max_new_tokens=256, do_sample=False)
            if isinstance(out, list) and out:
                text_out = out[0].get('generated_text', str(out[0]))
            else:
                text_out = str(out)

        # Clean common assistant/user prefixes that some models return verbatim
        if not isinstance(text_out, str):
            text_out = str(text_out)
        # strip leading role labels like 'Assistant:', 'Assistant -', 'User:' etc.
        cleaned = text_out.lstrip('\n\r ')
        for prefix in ("Assistant:", "Assistant -", "Assistant\n", "A:", "User:", "User -", "User\n"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].lstrip(' \n\r-:')
                break

        return cleaned.strip()
    except Exception as e:
        raise RuntimeError('Hugging Face model inference failed: ' + str(e))


@csrf_exempt
def send_message(request):
    """Endpoint that handles chat messages with conversation history."""
    if request.method != 'POST':
        return HttpResponseBadRequest('POST required')

    try:
        payload = json.loads(request.body.decode('utf-8'))
    except Exception:
        return HttpResponseBadRequest('Invalid JSON')
    
    text = payload.get('text', '').strip()
    session_id = payload.get('session_id', '')

    if not text:
        return HttpResponseBadRequest('text required')

    # Create or get session
    if not session_id:
        session_id = str(uuid.uuid4())
    
    try:
        # Use get_or_create to handle both new and existing sessions
        session, created = ConversationSession.objects.get_or_create(
            session_id=session_id,
            defaults={'user_agent': request.META.get('HTTP_USER_AGENT', '')}
        )
    except Exception as e:
        return JsonResponse({'ok': False, 'error': f'Session error: {str(e)}'}, status=500)

    # Store user message
    try:
        Message.objects.create(
            session=session,
            role='user',
            content=text
        )
    except Exception as e:
        return JsonResponse({'ok': False, 'error': f'Failed to save message: {str(e)}'}, status=500)

    # Get conversation history for context
    try:
        history = list(Message.objects.filter(session=session).values('role', 'content').order_by('created_at'))
    except Exception as e:
        history = []

    # Use Hugging Face model with conversation context (free)
    if not _HAS_HF:
        return JsonResponse({
            'ok': False,
            'error': 'Hugging Face transformers not installed. Install `transformers` and `torch` to enable local model answering.',
            'session_id': session_id
        }, status=503)

    try:
        answer = hf_answer(text, conversation_history=history)
        
        # Store assistant response
        Message.objects.create(
            session=session,
            role='assistant',
            content=answer
        )
        
        return JsonResponse({
            'ok': True,
            'reply': {'type': 'text', 'text': answer},
            'session_id': session_id
        })
    except Exception as e:
        return JsonResponse({
            'ok': False,
            'error': 'Local model inference failed: ' + str(e),
            'session_id': session_id
        }, status=500)


@csrf_exempt
def get_history(request):
    """Retrieve conversation history for a session."""
    if request.method != 'POST':
        return HttpResponseBadRequest('POST required')

    try:
        payload = json.loads(request.body.decode('utf-8'))
    except Exception:
        return HttpResponseBadRequest('Invalid JSON')
    
    session_id = payload.get('session_id', '')
    
    if not session_id:
        return JsonResponse({'ok': False, 'error': 'session_id required'}, status=400)
    
    try:
        session = ConversationSession.objects.get(session_id=session_id)
    except ConversationSession.DoesNotExist:
        # Return empty history if session doesn't exist
        return JsonResponse({
            'ok': True,
            'session_id': session_id,
            'messages': [],
            'note': 'Session does not exist yet'
        })
    
    # Retrieve all messages in the session
    try:
        messages = list(Message.objects.filter(session=session).values(
            'id', 'role', 'content', 'created_at'
        ).order_by('created_at'))
        
        return JsonResponse({
            'ok': True,
            'session_id': session_id,
            'messages': messages,
            'created_at': session.created_at.isoformat(),
            'updated_at': session.updated_at.isoformat()
        })
    except Exception as e:
        return JsonResponse({'ok': False, 'error': str(e)}, status=500)


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
