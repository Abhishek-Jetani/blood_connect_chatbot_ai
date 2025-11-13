#!/usr/bin/env python
"""
Comprehensive test script for Blood Connect Chatbot
Tests all endpoints and functionality
"""

import os
import sys
import json
import requests
import time
import uuid
from datetime import datetime

# Configuration
BASE_URL = os.getenv('CHATBOT_URL', 'http://localhost:8000/chatbot')
API_SEND = f'{BASE_URL}/api/send/'
API_HISTORY = f'{BASE_URL}/api/history/'
API_UPLOAD = f'{BASE_URL}/api/upload/'

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}{END}")
    print(f"{BLUE}{text.center(60)}{END}")
    print(f"{BLUE}{'='*60}{END}")

def print_success(text):
    print(f"{GREEN}✓ {text}{END}")

def print_error(text):
    print(f"{RED}✗ {text}{END}")

def print_info(text):
    print(f"{YELLOW}ℹ {text}{END}")

def print_result(key, value):
    print(f"  {BLUE}{key}:{END} {value}")


class ChatbotTester:
    def __init__(self):
        self.session_id = None
        self.test_results = []
        self.message_count = 0

    def test_connection(self):
        """Test if server is running"""
        print_header("Testing Server Connection")
        try:
            response = requests.get(f'{BASE_URL}/')
            if response.status_code == 200:
                print_success(f"Server is running on {BASE_URL}")
                return True
            else:
                print_error(f"Server returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print_error(f"Cannot connect to {BASE_URL}")
            print_info("Make sure Django server is running: python manage.py runserver")
            return False

    def test_send_message_without_session(self):
        """Test sending first message without session_id"""
        print_header("Test 1: Send Message Without Session ID")
        
        payload = {
            'text': 'Can I donate blood if I am under 18?'
        }
        print_info(f"Sending: {payload['text']}")
        print_result("Endpoint", API_SEND)
        
        try:
            start_time = time.time()
            response = requests.post(API_SEND, json=payload, timeout=30)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    self.session_id = data.get('session_id')
                    reply_text = data['reply']['text']
                    
                    print_success(f"Message sent and responded in {elapsed:.2f}s")
                    print_result("Session ID", self.session_id)
                    print_result("Reply", reply_text[:100] + "..." if len(reply_text) > 100 else reply_text)
                    self.message_count += 2  # user + assistant
                    self.test_results.append(('Send Message (No Session)', True))
                    return True
                else:
                    print_error(f"API returned error: {data.get('error')}")
                    self.test_results.append(('Send Message (No Session)', False))
                    return False
            else:
                print_error(f"HTTP {response.status_code}: {response.text}")
                self.test_results.append(('Send Message (No Session)', False))
                return False
        except Exception as e:
            print_error(f"Exception: {str(e)}")
            self.test_results.append(('Send Message (No Session)', False))
            return False

    def test_send_message_with_session(self):
        """Test sending message with existing session"""
        print_header("Test 2: Send Message With Existing Session ID")
        
        if not self.session_id:
            print_error("No session ID from previous test. Skipping.")
            return False
        
        payload = {
            'text': 'What blood types are accepted for donation?',
            'session_id': self.session_id
        }
        print_info(f"Sending: {payload['text']}")
        print_result("Session ID", self.session_id)
        
        try:
            start_time = time.time()
            response = requests.post(API_SEND, json=payload, timeout=30)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    reply_text = data['reply']['text']
                    
                    print_success(f"Message sent and responded in {elapsed:.2f}s")
                    print_result("Reply", reply_text[:100] + "..." if len(reply_text) > 100 else reply_text)
                    self.message_count += 2  # user + assistant
                    self.test_results.append(('Send Message (With Session)', True))
                    return True
                else:
                    print_error(f"API returned error: {data.get('error')}")
                    self.test_results.append(('Send Message (With Session)', False))
                    return False
            else:
                print_error(f"HTTP {response.status_code}: {response.text}")
                self.test_results.append(('Send Message (With Session)', False))
                return False
        except Exception as e:
            print_error(f"Exception: {str(e)}")
            self.test_results.append(('Send Message (With Session)', False))
            return False

    def test_context_awareness(self):
        """Test if model uses conversation context"""
        print_header("Test 3: Context Awareness")
        
        if not self.session_id:
            print_error("No session ID. Skipping.")
            return False
        
        payload = {
            'text': 'What is the minimum age?',
            'session_id': self.session_id
        }
        print_info("Testing if model remembers it's about blood donation...")
        print_info(f"Question: {payload['text']}")
        
        try:
            response = requests.post(API_SEND, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    reply_text = data['reply']['text'].lower()
                    # Check if response mentions blood or donation (context awareness)
                    context_keywords = ['blood', 'donate', 'donor', '18', 'age', 'year']
                    has_context = any(keyword in reply_text for keyword in context_keywords)
                    
                    if has_context:
                        print_success("Model appears to be context-aware")
                        print_result("Response", data['reply']['text'][:100] + "...")
                        self.message_count += 2
                        self.test_results.append(('Context Awareness', True))
                        return True
                    else:
                        print_info("Response may not show context awareness")
                        print_result("Response", data['reply']['text'][:100] + "...")
                        self.message_count += 2
                        self.test_results.append(('Context Awareness', False))
                        return False
                else:
                    print_error(f"API error: {data.get('error')}")
                    self.test_results.append(('Context Awareness', False))
                    return False
            else:
                print_error(f"HTTP {response.status_code}")
                self.test_results.append(('Context Awareness', False))
                return False
        except Exception as e:
            print_error(f"Exception: {str(e)}")
            self.test_results.append(('Context Awareness', False))
            return False

    def test_get_history(self):
        """Test retrieving conversation history"""
        print_header("Test 4: Get Conversation History")
        
        if not self.session_id:
            print_error("No session ID. Skipping.")
            return False
        
        payload = {'session_id': self.session_id}
        print_info(f"Retrieving history for session: {self.session_id}")
        
        try:
            response = requests.post(API_HISTORY, json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    messages = data.get('messages', [])
                    
                    print_success(f"Retrieved {len(messages)} messages")
                    print_result("Session Created", data['created_at'])
                    print_result("Last Updated", data['updated_at'])
                    
                    print(f"\n{BLUE}Message Timeline:{END}")
                    for i, msg in enumerate(messages[:5], 1):  # Show first 5
                        role = f"{BLUE}{msg['role'].upper()}{END}"
                        content = msg['content'][:60] + "..." if len(msg['content']) > 60 else msg['content']
                        print(f"  {i}. {role}: {content}")
                    
                    if len(messages) > 5:
                        print(f"  ... and {len(messages) - 5} more messages")
                    
                    self.test_results.append(('Get History', True))
                    return True
                else:
                    print_error(f"API error: {data.get('error')}")
                    self.test_results.append(('Get History', False))
                    return False
            else:
                print_error(f"HTTP {response.status_code}")
                self.test_results.append(('Get History', False))
                return False
        except Exception as e:
            print_error(f"Exception: {str(e)}")
            self.test_results.append(('Get History', False))
            return False

    def test_invalid_session(self):
        """Test with invalid session ID"""
        print_header("Test 5: Invalid Session ID Handling")
        
        invalid_session = str(uuid.uuid4())
        payload = {'session_id': invalid_session}
        
        print_info(f"Testing with invalid session: {invalid_session}")
        
        try:
            response = requests.post(API_HISTORY, json=payload, timeout=10)
            
            if response.status_code == 404:
                data = response.json()
                print_success("Server correctly rejected invalid session")
                print_result("Error message", data.get('error'))
                self.test_results.append(('Invalid Session Handling', True))
                return True
            else:
                print_error(f"Expected 404, got {response.status_code}")
                self.test_results.append(('Invalid Session Handling', False))
                return False
        except Exception as e:
            print_error(f"Exception: {str(e)}")
            self.test_results.append(('Invalid Session Handling', False))
            return False

    def test_empty_message(self):
        """Test sending empty message"""
        print_header("Test 6: Empty Message Handling")
        
        payload = {'text': '', 'session_id': str(uuid.uuid4())}
        print_info("Sending empty message...")
        
        try:
            response = requests.post(API_SEND, json=payload, timeout=10)
            
            if response.status_code == 400:
                print_success("Server correctly rejected empty message")
                self.test_results.append(('Empty Message Handling', True))
                return True
            else:
                print_error(f"Expected 400, got {response.status_code}")
                self.test_results.append(('Empty Message Handling', False))
                return False
        except Exception as e:
            print_error(f"Exception: {str(e)}")
            self.test_results.append(('Empty Message Handling', False))
            return False

    def test_multiple_sessions(self):
        """Test multiple independent sessions"""
        print_header("Test 7: Multiple Sessions")
        
        sessions = []
        print_info("Creating 3 independent sessions...")
        
        try:
            for i in range(3):
                payload = {'text': f'Question {i+1}'}
                response = requests.post(API_SEND, json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('ok'):
                        sessions.append(data['session_id'])
                        print_success(f"Session {i+1}: {data['session_id'][:12]}...")
                    else:
                        print_error(f"Session {i+1} failed: {data.get('error')}")
                        return False
                else:
                    print_error(f"Session {i+1}: HTTP {response.status_code}")
                    return False
            
            # Verify all sessions are different
            if len(sessions) == len(set(sessions)):
                print_success("All sessions have unique IDs")
                self.test_results.append(('Multiple Sessions', True))
                return True
            else:
                print_error("Duplicate session IDs detected!")
                self.test_results.append(('Multiple Sessions', False))
                return False
        except Exception as e:
            print_error(f"Exception: {str(e)}")
            self.test_results.append(('Multiple Sessions', False))
            return False

    def print_summary(self):
        """Print test summary"""
        print_header("Test Summary")
        
        passed = sum(1 for _, result in self.test_results if result)
        total = len(self.test_results)
        percentage = (passed / total * 100) if total > 0 else 0
        
        print(f"\n{BLUE}Results:{END}")
        for test_name, result in self.test_results:
            status = f"{GREEN}PASS{END}" if result else f"{RED}FAIL{END}"
            print(f"  {test_name}: {status}")
        
        print(f"\n{BLUE}Statistics:{END}")
        print_result("Tests Passed", f"{passed}/{total}")
        print_result("Success Rate", f"{percentage:.1f}%")
        print_result("Total Messages Stored", self.message_count)
        
        if percentage == 100:
            print_success("All tests passed! ✓")
        elif percentage >= 70:
            print_info("Most tests passed. Check failures above.")
        else:
            print_error("Multiple tests failed. Check setup.")

    def run_all_tests(self):
        """Run all tests"""
        print_header("Blood Connect Chatbot - Test Suite")
        print(f"Testing URL: {BASE_URL}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run tests in order
        if not self.test_connection():
            print_error("Cannot proceed - server not running")
            return False
        
        self.test_send_message_without_session()
        self.test_send_message_with_session()
        self.test_context_awareness()
        self.test_get_history()
        self.test_invalid_session()
        self.test_empty_message()
        self.test_multiple_sessions()
        
        self.print_summary()
        
        return True


if __name__ == '__main__':
    tester = ChatbotTester()
    tester.run_all_tests()
