import json
import os
from pathlib import Path
from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# Runtime LLM integration (OpenAI-compatible). No static FAQ pairs — answers come from the model.
import os

# Try to use a local Hugging Face model (free) via transformers if available.
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    _HAS_HF = True
except Exception:
    pipeline = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    _HAS_HF = False


def chat_ui(request):
    """Render the chat UI template."""
    return render(request, 'chatbot/chat.html')


# Runtime LLM caller (OpenAI-compatible). Configure with OPENAI_API_KEY in environment.
# No external API calls here — prefer local Hugging Face models only (free).


def hf_answer(question):
    """Use a local Hugging Face seq2seq model to answer the question.

    Set HUGGINGFACE_MODEL env var to choose model (default: google/flan-t5-small).
    Requires `transformers` and `torch` installed. Loads model on first call.
    """
    if not _HAS_HF:
        raise RuntimeError('Hugging Face transformers not available')

    model_name = os.environ.get('HUGGINGFACE_MODEL', 'google/flan-t5-small')
    # cache pipeline in module attribute
    if not hasattr(hf_answer, 'pipe') or hf_answer.pipe is None:
        # load model and tokenizer
        try:
            hf_answer.pipe = pipeline('text2text-generation', model=model_name, device=-1)
        except Exception as e:
            # try manual model/tokenizer load for some environments
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                hf_answer.pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=-1)
            except Exception as e2:
                raise RuntimeError(f'Failed to load Hugging Face model {model_name}: {e2}')

    prompt = (
        "Answer concisely and accurately about blood donation. If urgent, advise contacting medical professionals.\nQuestion: "
        + question
    )
    try:
        out = hf_answer.pipe(prompt, max_length=256, do_sample=False)
        if isinstance(out, list) and out:
            return out[0].get('generated_text', str(out[0]))
        return str(out)
    except Exception as e:
        raise RuntimeError('Hugging Face model inference failed: ' + str(e))


@csrf_exempt
def send_message(request):
    """Endpoint that handles quick replies and a lightweight FAQ-based ML answerer."""
    if request.method != 'POST':
        return HttpResponseBadRequest('POST required')

    try:
        payload = json.loads(request.body.decode('utf-8'))
    except Exception:
        return HttpResponseBadRequest('Invalid JSON')
    # Always take the user's query from the 'text' field and answer via the ML model.
    text = payload.get('text', '')

    if not text:
        return HttpResponseBadRequest('text required')

    # Use only a local Hugging Face model (free). If not available, return a helpful error.
    if not _HAS_HF:
        return JsonResponse({'ok': False, 'error': 'Hugging Face transformers not installed. Install `transformers` and `torch` to enable local model answering.'}, status=503)

    try:
        answer = hf_answer(text)
        return JsonResponse({'ok': True, 'reply': {'type': 'text', 'text': answer}})
    except Exception as e:
        return JsonResponse({'ok': False, 'error': 'Local model inference failed: ' + str(e)}, status=500)


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
