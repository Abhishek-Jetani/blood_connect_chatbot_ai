#!/usr/bin/env python
"""
Debug and test script for Blood Donation Chatbot
Run this to verify the AI models are working correctly
"""

import os
import sys
import django
import logging

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
django.setup()

from chatbot.blood_assistant import get_assistant, answer_question

# Configure logging for visibility
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_blood_assistant():
    """Test the blood assistant with various questions"""
    
    print("\n" + "="*80)
    print("BLOOD DONATION CHATBOT - DEBUG TEST")
    print("="*80 + "\n")
    
    # Initialize assistant
    print("Initializing Blood Assistant...")
    try:
        assistant = get_assistant()
        print("✓ Blood Assistant initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize Blood Assistant: {e}")
        return
    
    # Test questions covering different aspects
    test_cases = [
        {
            "question": "Can I donate blood? I'm 25 years old and healthy.",
            "description": "Eligibility question"
        },
        {
            "question": "What should I eat and drink before donating?",
            "description": "Preparation question"
        },
        {
            "question": "How long does the whole donation process take?",
            "description": "Process duration question"
        },
        {
            "question": "What are the benefits of blood donation?",
            "description": "Benefits question"
        },
        {
            "question": "What should I do after donating blood?",
            "description": "Recovery/aftercare question"
        },
        {
            "question": "I feel dizzy after donating, is this normal?",
            "description": "Side effects question"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        description = test_case["description"]
        
        print("\n" + "-"*80)
        print(f"TEST {i}/{len(test_cases)}: {description}")
        print("-"*80)
        print(f"Q: {question}\n")
        
        try:
            result = answer_question(question)
            
            print(f"A: {result['answer']}\n")
            print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2%})")
            print(f"Sources retrieved: {len(result['sources'])}")
            
            if result['sources']:
                print("Relevant documents:")
                for j, source in enumerate(result['sources'], 1):
                    print(f"  {j}. {source[:70]}...")
            
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            print(f"✗ Error: {e}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")
    
    # Test conversation history
    print("Testing with conversation history...\n")
    conversation = []
    
    questions_with_history = [
        "I want to donate blood.",
        "What are the requirements?",
        "What should I eat before?",
        "How long will it take?"
    ]
    
    for question in questions_with_history:
        print(f"Q: {question}")
        result = answer_question(question, conversation_history=conversation)
        print(f"A: {result['answer']}\n")
        
        # Add to conversation
        conversation.append({'role': 'user', 'content': question})
        conversation.append({'role': 'assistant', 'content': result['answer']})


def model_info():
    """Print information about loaded models"""
    print("\n" + "="*80)
    print("MODEL INFORMATION")
    print("="*80 + "\n")
    
    try:
        assistant = get_assistant()
        
        print("✓ Models Successfully Loaded:")
        for model_type, model_name in assistant.models_loaded.items():
            print(f"  • {model_type}: {model_name}")
        
        print(f"\n✓ Device: {assistant.device.upper()}")
        
        if hasattr(assistant, 'knowledge_texts'):
            print(f"✓ Knowledge base entries: {len(assistant.knowledge_texts)}")
        
        print(f"\n✓ All required libraries available!")
        
    except Exception as e:
        print(f"✗ Error getting model info: {e}")
        logger.error("Model info error", exc_info=True)


if __name__ == "__main__":
    import sys
    
    print("\nWelcome to Blood Donation Chatbot Debugger")
    print("=" * 80)
    print("\nOptions:")
    print("  1. Run full test suite (recommended)")
    print("  2. Show model information")
    print("  3. Interactive mode (ask questions)")
    print("  4. All of the above\n")
    
    choice = input("Enter your choice (1-4) or press Enter for 1: ").strip() or "1"
    
    if choice in ("1", "4"):
        test_blood_assistant()
    
    if choice in ("2", "4"):
        model_info()
    
    if choice == "3":
        print("\n" + "="*80)
        print("INTERACTIVE MODE")
        print("="*80)
        print("Type 'quit' to exit\n")
        
        try:
            assistant = get_assistant()
            conversation = []
            
            while True:
                question = input("\nYou: ").strip()
                
                if question.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                result = answer_question(question, conversation_history=conversation)
                print(f"\nAssistant: {result['answer']}")
                print(f"(Intent: {result['intent']}, Confidence: {result['confidence']:.2%})")
                
                conversation.append({'role': 'user', 'content': question})
                conversation.append({'role': 'assistant', 'content': result['answer']})
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            logger.error("Interactive mode error", exc_info=True)
            print(f"Error: {e}")
