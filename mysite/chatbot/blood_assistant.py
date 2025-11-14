"""
Blood Donation Chatbot Assistant using Free, Open-Source Models
Uses DistilBERT and text generation for medical Q&A

Models Used:
1. distilbert-base-uncased-finetuned-sst-2-english - Sentiment/Intent Classification
2. google/flan-t5-small - Free text generation model (no API key needed)
3. sentence-transformers/all-MiniLM-L6-v2 - Semantic similarity for retrieval

All models are open-source and free to use locally!
"""

import logging
import os
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime

# Configure logging for debugging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Try to import required libraries
try:
    from transformers import (
        pipeline,
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        DistilBertForSequenceClassification,
        DistilBertTokenizer
    )
    logger.info("✓ Transformers library loaded successfully")
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger.error(f"✗ Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

try:
    import torch
    logger.info(f"✓ PyTorch loaded (GPU available: {torch.cuda.is_available()})")
    TORCH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠ PyTorch not available: {e}")
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    logger.info("✓ Sentence-transformers loaded for semantic search")
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠ Sentence-transformers not available: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class BloodAssistant:
    """
    Blood Donation Chatbot Assistant with multiple free models
    
    Features:
    - Intent classification using DistilBERT
    - Answer generation using FLAN-T5
    - Semantic search for knowledge base
    - Full debugging and logging
    """
    
    # Blood donation knowledge base
    KNOWLEDGE_BASE = {
        "eligibility": {
            "age": "Blood donors must be between 18-65 years old",
            "weight": "Donors should weigh at least 50 kg (110 lbs)",
            "health": "Must be in good health without active infections",
            "medications": "Certain medications may disqualify you from donating"
        },
        "process": {
            "screening": "Complete medical history screening (5-10 minutes)",
            "physical": "Physical exam including blood pressure and temperature check",
            "donation": "Actual blood donation takes 5-10 minutes",
            "recovery": "Refreshments provided, rest for 10-15 minutes after"
        },
        "benefits": {
            "help": "Your donation can save up to 3 lives",
            "health": "Free health screening during donation",
            "community": "Help your local community in emergencies",
            "rewards": "Many centers offer rewards or incentives"
        },
        "preparation": {
            "hydration": "Drink plenty of water 24 hours before donation",
            "food": "Eat a healthy meal 2-3 hours before donating",
            "sleep": "Get adequate sleep the night before",
            "avoid": "Avoid alcohol, smoking, and strenuous exercise before donation"
        },
        "recovery": {
            "fluids": "Drink extra fluids for 48 hours after donation",
            "rest": "Rest for at least 24 hours after donation",
            "food": "Eat iron-rich foods like spinach and red meat",
            "activity": "Avoid strenuous activities for 24 hours"
        }
    }
    
    def __init__(self):
        """Initialize the Blood Assistant with models"""
        logger.info("=" * 70)
        logger.info("INITIALIZING BLOOD DONATION CHATBOT ASSISTANT")
        logger.info("=" * 70)
        
        self.models_loaded = {}
        self.device = self._get_device()
        
        # Initialize model attributes as None
        self.text_gen_pipeline = None
        self.intent_classifier = None
        self.semantic_model = None
        self.knowledge_texts = []
        self.knowledge_embeddings = None
        
        # Initialize models
        if TRANSFORMERS_AVAILABLE:
            self._init_text_generation_model()
            self._init_intent_classifier()
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self._init_semantic_search_model()
        
        logger.info("=" * 70)
        logger.info("INITIALIZATION COMPLETE")
        logger.info(f"Models loaded: {list(self.models_loaded.keys())}")
        logger.info("=" * 70)
    
    def _get_device(self) -> str:
        """Determine whether to use GPU or CPU"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device = "cuda"
            logger.info(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("✓ Using CPU (GPU not available)")
        return device
    
    def _init_text_generation_model(self):
        """Initialize FLAN-T5 model for text generation"""
        try:
            model_name = os.environ.get('HUGGINGFACE_MODEL', 'google/flan-t5-small')
            logger.info(f"Loading text generation model: {model_name}")
            
            device_id = 0 if self.device == "cuda" else -1
            
            self.text_gen_pipeline = pipeline(
                "text2text-generation",
                model=model_name,
                device=device_id,
                trust_remote_code=True
            )
            
            self.models_loaded['text_generation'] = model_name
            logger.info(f"✓ Text generation model loaded: {model_name}")
        except Exception as e:
            logger.error(f"✗ Failed to load text generation model: {e}")
            self.text_gen_pipeline = None
    
    def _init_intent_classifier(self):
        """Initialize DistilBERT for intent classification"""
        try:
            model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
            logger.info(f"Loading intent classifier: {model_name}")
            
            device_id = 0 if self.device == "cuda" else -1
            
            self.intent_classifier = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=device_id,
                trust_remote_code=True
            )
            
            self.models_loaded['intent_classifier'] = model_name
            logger.info(f"✓ Intent classifier loaded: {model_name}")
        except Exception as e:
            logger.error(f"✗ Failed to load intent classifier: {e}")
            self.intent_classifier = None
    
    def _init_semantic_search_model(self):
        """Initialize Sentence Transformer for semantic search"""
        try:
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
            logger.info(f"Loading semantic search model: {model_name}")
            
            device = "cuda" if self.device == "cuda" else "cpu"
            self.semantic_model = SentenceTransformer(model_name, device=device)
            
            # Encode knowledge base
            self._encode_knowledge_base()
            
            self.models_loaded['semantic_search'] = model_name
            logger.info(f"✓ Semantic search model loaded: {model_name}")
        except Exception as e:
            logger.error(f"✗ Failed to load semantic search model: {e}")
            self.semantic_model = None
            self.knowledge_embeddings = None
    
    def _encode_knowledge_base(self):
        """Encode knowledge base for semantic search"""
        try:
            logger.debug("Encoding knowledge base for semantic search...")
            
            # Flatten knowledge base into texts
            self.knowledge_texts = []
            for category, items in self.KNOWLEDGE_BASE.items():
                for key, value in items.items():
                    self.knowledge_texts.append(f"{category} - {key}: {value}")
            
            # Encode all texts
            self.knowledge_embeddings = self.semantic_model.encode(
                self.knowledge_texts,
                convert_to_tensor=True
            )
            
            logger.info(f"✓ Encoded {len(self.knowledge_texts)} knowledge base entries")
        except Exception as e:
            logger.error(f"✗ Failed to encode knowledge base: {e}")
            self.knowledge_texts = []
            self.knowledge_embeddings = None
    
    def classify_intent(self, question: str) -> Dict:
        """Classify the intent of user's question"""
        logger.debug(f"Classifying intent for: {question[:100]}")
        
        if not self.intent_classifier:
            logger.warning("Intent classifier not available")
            return {"intent": "general", "confidence": 0.0}
        
        try:
            result = self.intent_classifier(question[:512])[0]
            intent_map = {
                'POSITIVE': 'interested',
                'NEGATIVE': 'concerned'
            }
            
            intent_result = {
                "intent": intent_map.get(result['label'], 'neutral'),
                "confidence": result['score']
            }
            
            logger.debug(f"Intent: {intent_result['intent']} (confidence: {intent_result['confidence']:.2f})")
            return intent_result
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return {"intent": "general", "confidence": 0.0}
    
    def retrieve_relevant_docs(self, question: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant documents from knowledge base using semantic search"""
        logger.debug(f"Retrieving relevant docs for: {question[:100]}")
        
        if not self.semantic_model or not self.knowledge_embeddings:
            logger.warning("Semantic search not available")
            return []
        
        try:
            # Encode question
            question_embedding = self.semantic_model.encode(question, convert_to_tensor=True)
            
            # Find similar documents
            similarities = util.pytorch_cos_sim(question_embedding, self.knowledge_embeddings)[0]

            # Get top-k results
            topk = min(top_k, len(similarities))
            top_results = torch.topk(similarities, k=topk)

            relevant_docs = []
            for score, idx in zip(top_results[0], top_results[1]):
                doc = self.knowledge_texts[idx.item()]
                sc = float(score.item())
                relevant_docs.append((doc, sc))
                logger.debug(f"  - {doc[:80]} (similarity: {sc:.3f})")

            return relevant_docs
        except Exception as e:
            logger.error(f"Document retrieval error: {e}")
            return []

    def _is_contact_request(self, question: str) -> bool:
        """Detect whether the user is asking for local contact/phone numbers for blood banks."""
        if not question:
            return False
        q = question.lower()
        contact_keywords = ["contact", "contact number", "phone", "phone number", "phone no", "call", "telephone", "helpline", "number"]
        # check for blood/blood bank + a contact keyword
        is_blood = "blood" in q or "blood bank" in q or "bloodbank" in q
        has_contact_kw = any(k in q for k in contact_keywords)
        return is_blood and has_contact_kw

    def _handle_faq(self, question: str) -> Optional[str]:
        """Handle short, common FAQ-style questions with deterministic answers.

        This prevents the generator from hallucinating for ambiguous, high-value
        questions such as "Which blood group is good?" by returning a concise,
        evidence-based response or a clarifying question.
        """
        if not question:
            return None

        q = question.lower().strip()

        # Common ambiguous / FAQ patterns
        faq_patterns = [
            "which blood group is good",
            "which blood group is best",
            "best blood group",
            "which blood group should i",
            "which blood group is rare",
            "which blood group is universal",
            "which blood group to donate",
        ]

        if any(p in q for p in faq_patterns):
            return (
                "No blood group is universally \"good\" — it depends on context. "
                "For transfusions, compatibility matters: O-negative is the universal red-cell donor, "
                "AB-positive is the universal plasma recipient, and specific matches are used when possible. "
                "All blood types are valuable for donation. Do you mean which is best for donating, "
                "for receiving transfusions, or for general health?"
            )

        return None

    def _handle_out_of_domain(self, question: str) -> Optional[str]:
        """Detect and handle questions that are outside the assistant's domain.

        Returns a short redirect/clarifying response instead of invoking the generator
        which can accidentally echo the assistant's system-role text.
        """
        if not question:
            return None

        q = question.lower()

        # Patterns that indicate current-affairs / political / general-knowledge
        general_kw = [
            "prime minister",
            "pm of",
            "who is the pm",
            "who is the prime minister",
            "who is the president",
            "capital of",
            "population of",
            "who is the king",
            "current",
            "what is the pm",
        ]

        if any(k in q for k in general_kw):
            return (
                "I specialise in blood donation information and related medical guidance. "
                "I don't provide up-to-date general knowledge or current-affairs facts. "
                "For the current Prime Minister of India, please check reliable news sources, "
                "the official government website, or Wikipedia. If you want, I can try to answer "
                "but my information may be out of date."
            )

        return None

    def _is_in_domain(self, question: str) -> Tuple[bool, float]:
        """Determine whether a question is related to blood donation.

        Returns (in_domain: bool, score: float). Uses the semantic search model
        and knowledge base embeddings when available. Falls back to a keyword
        check if semantic model isn't available.
        """
        if not question:
            return False, 0.0

        q = question.strip()

        # If we have semantic search available, use cosine similarity to knowledge base
        try:
            if self.semantic_model and self.knowledge_embeddings is not None and len(self.knowledge_embeddings) > 0:
                question_embedding = self.semantic_model.encode(q, convert_to_tensor=True)
                sims = util.pytorch_cos_sim(question_embedding, self.knowledge_embeddings)[0]
                # Get max similarity score
                max_score = float(torch.max(sims).item())
                # Threshold can be tuned; make overridable via env var
                try:
                    threshold = float(os.environ.get('DOMAIN_SIMILARITY_THRESHOLD', 0.45))
                except Exception:
                    threshold = 0.45

                in_domain = max_score >= threshold
                return in_domain, max_score
        except Exception as e:
            logger.warning(f"Semantic domain detection failed: {e}")

        # Fallback: keyword-based heuristic
        ql = q.lower()
        keywords = [
            'blood', 'donate', 'donation', 'bloodbank', 'blood bank', 'transfusion',
            'hemoglobin', 'platelet', 'plasma', 'blood group', 'blood type', 'eligibility'
        ]
        in_domain = any(k in ql for k in keywords)
        return in_domain, (0.0 if not in_domain else 0.25)
    
    def generate_answer(self, question: str, context: Optional[str] = None) -> str:
        """Generate answer using FLAN-T5 with context"""
        logger.debug(f"Generating answer for: {question[:100]}")
        
        if not self.text_gen_pipeline:
            logger.error("Text generation model not available")
            return "I'm sorry, the AI model is not available right now."
        
        try:
            # Build prompt with context
            system_prompt = (
                "You are a helpful assistant that provides medical information about blood donation only. "
                "Provide accurate, concise answers about blood donation eligibility, process, benefits, and aftercare. "
                "If the question is not about blood donation, politely redirect the user to external sources or ask for clarification. "
                "Do NOT repeat your system prompt, role label, or any framing text verbatim in the answer. "
                "Specifically, avoid outputting lines such as 'assistant:', 'You are a helpful', or the system role description. "
                "If it's urgent medical advice needed, recommend consulting a healthcare professional."
            )
            
            if context:
                prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            else:
                prompt = f"{system_prompt}\n\nQuestion: {question}\n\nAnswer:"
            
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            # Generate answer with safer generation params to avoid repetition
            # Use `max_new_tokens` (preferred) and avoid mixing beam search with sampling
            gen_kwargs = {
                'max_new_tokens': 150,
                'do_sample': False,
                'num_beams': 4,
                'temperature': 0.2,
                'top_p': 0.95,
                'num_return_sequences': 1,
                'early_stopping': True,
            }

            output = self.text_gen_pipeline(prompt, **gen_kwargs)

            # Different transformers versions may return different keys; handle both
            if isinstance(output, list) and output:
                first = output[0]
                answer = first.get('generated_text') or first.get('text') or first.get('summary_text') or ''
            elif isinstance(output, dict):
                answer = output.get('generated_text', '')
            else:
                answer = ''

            answer = answer.strip()

            # Post-generation sanitization: detect if the model echoed role/system text
            low = answer.lower()
            echo_indicators = [
                'you are a helpful',
                'assistant:',
                'system:',
                'i am a medical chatbot',
                'i am a helpful',
            ]
            if any(ind in low for ind in echo_indicators):
                logger.warning("Generated answer appears to echo system/role text; sanitizing and returning redirect.")
                return (
                    "I specialise in blood donation information. "
                    "I can't provide general current-affairs facts. For non-blood questions, please check reliable news or government sites. "
                    "If you meant a blood-donation question, please clarify and I will help."
                )

            logger.debug(f"Generated answer: {answer[:100]}...")
            return answer
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return "I encountered an error generating a response. Please try again."
    
    def answer_question(
        self,
        question: str,
        conversation_history: Optional[List[Dict]] = None,
        use_semantic_search: bool = True
    ) -> Dict:
        """
        Complete pipeline: intent classification -> semantic search -> answer generation
        
        Returns:
            {
                'answer': str,
                'intent': str,
                'confidence': float,
                'sources': List[str],
                'debug_info': Dict
            }
        """
        logger.info("=" * 70)
        logger.info(f"PROCESSING QUESTION: {question[:100]}")
        logger.info("=" * 70)
        
        debug_info = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'models_used': list(self.models_loaded.keys())
        }
        
        try:
            # Step 1: Classify intent
            logger.info("STEP 1: Classifying intent...")
            intent_result = self.classify_intent(question)
            debug_info['intent'] = intent_result

            # Dynamic domain detection: determine if the question is about blood donation
            in_domain, domain_score = self._is_in_domain(question)
            debug_info['domain_score'] = domain_score
            debug_info['in_domain'] = in_domain

            # If not in domain, return a short redirect instead of invoking the generator
            if not in_domain:
                out_of_domain_answer = self._handle_out_of_domain(question)
                debug_info['out_of_domain_handled'] = True
                return {
                    'answer': out_of_domain_answer,
                    'intent': intent_result['intent'],
                    'confidence': intent_result['confidence'],
                    'sources': [],
                    'debug_info': debug_info,
                }

            # For short/ambiguous but in-domain FAQs, return a deterministic FAQ reply
            faq_answer = self._handle_faq(question)
            if faq_answer:
                debug_info['faq_handled'] = True
                return {
                    'answer': faq_answer,
                    'intent': intent_result['intent'],
                    'confidence': intent_result['confidence'],
                    'sources': [],
                    'debug_info': debug_info,
                }
            
            # Step 2: Retrieve relevant documents
            logger.info("STEP 2: Retrieving relevant documents...")
            relevant_docs = []
            if use_semantic_search:
                retrieved = self.retrieve_relevant_docs(question, top_k=6)
                # retrieve_relevant_docs now returns list of (doc, score)
                # filter low-similarity docs to avoid adding noisy context
                filtered = [d for d, s in retrieved if s >= float(os.environ.get('CONTEXT_MIN_SIM', 0.35))]
                relevant_docs = filtered[:3]
                debug_info['retrieved_raw'] = retrieved
                debug_info['retrieved_docs'] = relevant_docs
            else:
                debug_info['retrieved_docs'] = []

            # If user asks specifically for local contact numbers and we have no contact DB,
            # return a helpful, actionable message instead of relying on the generator.
            if not relevant_docs and self._is_contact_request(question):
                logger.info("No local contact data available for contact request — returning guidance.")
                canned = (
                    "I don't have a local contacts database of blood banks. "
                    "To find blood bank contact numbers in Gujarat, you can:\n"
                    "1) Search Google Maps for 'blood bank' plus the city name (e.g., 'blood bank Ahmedabad').\n"
                    "2) Check the Gujarat Health Department or National Blood Transfusion Service websites.\n"
                    "3) Contact nearby hospitals or the Indian Red Cross Society for emergency help.\n"
                    "If you provide a specific city in Gujarat I can suggest search terms or help format a short list you can use locally."
                )

                return {
                    'answer': canned,
                    'intent': debug_info.get('intent', {}).get('intent', 'general'),
                    'confidence': debug_info.get('intent', {}).get('confidence', 0.0),
                    'sources': [],
                    'debug_info': debug_info,
                }
            
            # Step 3: Build context
            context = ""
            if relevant_docs:
                # relevant_docs is a list of strings after filtering
                context = "\n".join(relevant_docs)
            
            if conversation_history:
                # Add recent conversation history but avoid using the literal token 'assistant'
                # since short prompts can cause the generator to echo role text.
                recent_msgs = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
                formatted_history = []
                for m in recent_msgs:
                    if m.get('role') == 'user':
                        formatted_history.append(f"User said: {m.get('content')}")
                    else:
                        formatted_history.append(f"Previous reply: {m.get('content')}")

                history_context = "\n".join(formatted_history)
                context = f"{history_context}\n\n{context}" if context else history_context
                debug_info['history_included'] = True
            
            # Step 4: Generate answer
            logger.info("STEP 3: Generating answer...")
            answer = self.generate_answer(question, context)
            
            result = {
                'answer': answer,
                'intent': intent_result['intent'],
                'confidence': intent_result['confidence'],
                'sources': relevant_docs,
                'debug_info': debug_info
            }
            
            logger.info("=" * 70)
            logger.info("QUESTION PROCESSING COMPLETE")
            logger.info(f"Answer preview: {answer[:100]}...")
            logger.info("=" * 70)
            
            return result
        
        except Exception as e:
            logger.error(f"Error in answer_question pipeline: {e}", exc_info=True)
            return {
                'answer': f"An error occurred: {str(e)}",
                'intent': 'error',
                'confidence': 0.0,
                'sources': [],
                'debug_info': {**debug_info, 'error': str(e)}
            }


# Global instance
_assistant_instance = None


def get_assistant() -> BloodAssistant:
    """Get or create the blood assistant singleton"""
    global _assistant_instance
    if _assistant_instance is None:
        logger.info("Creating new BloodAssistant instance")
        _assistant_instance = BloodAssistant()
    return _assistant_instance


def answer_question(
    question: str,
    conversation_history: Optional[List[Dict]] = None,
    debug: bool = True
) -> Dict:
    """
    Public API for answering questions
    
    Args:
        question: User's question
        conversation_history: Previous messages for context
        debug: Whether to include debug info in response
    
    Returns:
        Response dictionary with answer and metadata
    """
    if debug:
        logger.info(f"\n{'='*70}")
        logger.info(f"API CALL: answer_question()")
        logger.info(f"  Question: {question[:100]}")
        logger.info(f"  History length: {len(conversation_history) if conversation_history else 0}")
        logger.info(f"{'='*70}\n")
    
    assistant = get_assistant()
    return assistant.answer_question(
        question,
        conversation_history=conversation_history,
        use_semantic_search=True
    )


if __name__ == "__main__":
    """Test the assistant"""
    logger.info("Starting Blood Assistant test...")
    
    assistant = get_assistant()
    
    # Test questions
    test_questions = [
        "Can I donate blood? I'm 25 years old and healthy.",
        "What should I eat before donating blood?",
        "How long does the donation process take?",
        "What are the benefits of donating blood?",
        "What should I do after donating blood?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"TEST {i}/5")
        logger.info(f"{'='*70}")
        
        result = assistant.answer_question(question)
        
        logger.info(f"Answer: {result['answer']}")
        logger.info(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        logger.info(f"Sources: {len(result['sources'])} documents retrieved")
