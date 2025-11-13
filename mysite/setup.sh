#!/bin/bash
# Blood Connect Chatbot - One Command Setup
# Run this script to setup everything at once

echo "🚀 Blood Connect Chatbot - Setup Script"
echo "========================================"
echo ""

# Check if in correct directory
if [ ! -f "manage.py" ]; then
    echo "❌ Error: manage.py not found!"
    echo "Make sure you run this from the mysite directory"
    exit 1
fi

# Step 1: Install dependencies
echo "📦 Step 1: Installing dependencies..."
pip install -r requirements.txt > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Step 2: Create migrations
echo "📝 Step 2: Creating migrations..."
python manage.py makemigrations chatbot > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Migrations created"
else
    echo "❌ Failed to create migrations"
    exit 1
fi

# Step 3: Apply migrations
echo "🔄 Step 3: Applying migrations..."
python manage.py migrate > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Database tables created"
else
    echo "❌ Failed to apply migrations"
    exit 1
fi

# Step 4: Verify setup
echo "✔️  Step 4: Verifying setup..."
python manage.py check > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Setup verification passed"
else
    echo "❌ Setup verification failed"
    exit 1
fi

echo ""
echo "✨ Setup Complete! ✨"
echo ""
echo "Next steps:"
echo "  1. Run: python manage.py runserver"
echo "  2. Open: http://localhost:8000/chatbot/"
echo "  3. Start chatting!"
echo ""
echo "Need help? Read START_HERE.md"
