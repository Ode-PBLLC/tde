#!/usr/bin/env python3
"""
Simple script to check Anthropic API status before running the demo.
"""

import asyncio
import os
import sys
from datetime import datetime

# Add the mcp directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mcp'))

async def check_api_status():
    """Check if Anthropic API is responsive."""
    try:
        print("🔍 Checking Anthropic API status...")
        
        # Import after path setup
        import anthropic
        
        # Check if API key is set
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("❌ ANTHROPIC_API_KEY environment variable not set")
            return False
        
        print(f"✅ API key found (length: {len(api_key)})")
        
        # Test API with minimal request
        client = anthropic.Anthropic()
        
        print("🔧 Testing API connection...")
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        
        if response and response.content:
            print("✅ API is responsive and working")
            print(f"📊 Response received at {datetime.now().strftime('%H:%M:%S')}")
            return True
        else:
            print("❌ API responded but with empty content")
            return False
            
    except Exception as e:
        error_msg = str(e)
        
        if "overloaded" in error_msg.lower() or "529" in error_msg:
            print("⏳ API is currently overloaded (Error 529)")
            print("💡 Try again in 2-5 minutes")
            return False
        elif "401" in error_msg or "authentication" in error_msg.lower():
            print("🔑 API key authentication failed")
            print("💡 Check your ANTHROPIC_API_KEY")
            return False
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            print("💳 API quota/rate limit exceeded")
            print("💡 Wait or check your usage limits")
            return False
        else:
            print(f"❌ API error: {error_msg}")
            return False

async def main():
    """Main function to check API and provide guidance."""
    print("🚀 Anthropic API Status Checker")
    print("=" * 40)
    
    # Check API status
    api_ok = await check_api_status()
    
    print("\n" + "=" * 40)
    
    if api_ok:
        print("🎉 API is working! You can run the citation demo:")
        print("   python run_citation_demo.py")
        print("   OR")
        print("   streamlit run streamlit_citation_demo.py")
    else:
        print("⚠️  API not available right now.")
        print("\n🔧 Troubleshooting steps:")
        print("1. Wait 2-5 minutes if overloaded (most common)")
        print("2. Check ANTHROPIC_API_KEY environment variable")
        print("3. Verify API key has quota/credits")
        print("4. Check https://status.anthropic.com for outages")
        print("\n🔄 Run this script again to recheck")

if __name__ == "__main__":
    asyncio.run(main())