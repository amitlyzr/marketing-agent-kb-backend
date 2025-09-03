#!/usr/bin/env python3
"""
Test script for text training with chat history format
"""

import json
from pdf_utils import train_text_directly, process_completed_interview

def test_text_training_with_sample_data():
    """
    Test the text training function with sample chat history data
    """
    
    # Sample chat history in the format you provided
    sample_chat_history = [
        {
            "role": "user",
            "content": "hello",
            "created_at": "2025-09-01T17:43:12.540000"
        },
        {
            "role": "assistant",
            "content": "Hello! Thank you for reaching out. How can I assist you today? If you're here for an interview, I'd be happy to get started. Please tell me a bit about your background and what role you're interviewing for. This will help me tailor my questions to assess your skills and experience effectively.",
            "created_at": "2025-09-01T17:43:12.545000"
        },
        {
            "role": "user",
            "content": "I'm a software engineer with 5 years of experience",
            "created_at": "2025-09-01T17:43:30.123000"
        },
        {
            "role": "assistant",
            "content": "Great! Can you tell me about a challenging project you worked on recently?",
            "created_at": "2025-09-01T17:43:35.456000"
        }
    ]
    
    print("🧪 Testing Text Training with Sample Chat History")
    print("=" * 60)
    
    # Convert chat history to text format (same as in process_completed_interview)
    text_content = ""
    for message in sample_chat_history:
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        created_at = message.get('created_at', '')
        
        text_content += f"[{role.upper()}] {content}\n"
        if created_at:
            text_content += f"Time: {created_at}\n"
        text_content += "\n"
    
    print("📝 Generated Text Content:")
    print("-" * 30)
    print(text_content)
    print("-" * 30)
    
    print(f"📊 Text Statistics:")
    print(f"   - Character count: {len(text_content)}")
    print(f"   - Word count: {len(text_content.split())}")
    print(f"   - Line count: {len(text_content.split(chr(10)))}")
    print(f"   - Messages: {len(sample_chat_history)}")
    
    # Test parameters
    test_rag_id = "68b721bc660680eb1bfbfe4b"  # Your test RAG ID
    test_api_key = "sk-default-AP2dc2OE8ElziiXisioG75rOg6EZZ8B8"  # Your test API key
    
    print(f"\n🔧 Test Configuration:")
    print(f"   - RAG ID: {test_rag_id}")
    print(f"   - API Key: {test_api_key[:15]}...")
    print(f"   - Data Parser: simple")
    print(f"   - Chunk Size: 1000")
    print(f"   - Chunk Overlap: 100")
    
    try:
        print(f"\n🚀 Starting Text Training...")
        
        # Call the text training function
        result = train_text_directly(
            text_content=text_content,
            rag_id=test_rag_id,
            api_key=test_api_key,
            data_parser="simple",
            chunk_size=1000,
            chunk_overlap=100,
            extra_info="{}"
        )
        
        print(f"✅ Text Training Completed Successfully!")
        print(f"📊 Training Result:")
        print(json.dumps(result, indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ Text Training Failed: {e}")
        print(f"Error Type: {type(e).__name__}")
        return False

def test_full_workflow():
    """
    Test the complete process_completed_interview workflow
    """
    print(f"\n" + "=" * 60)
    print("🧪 Testing Complete Interview Processing Workflow")
    print("=" * 60)
    
    # Test parameters
    test_user_id = "mem_cme8pevbm00hi0wmc3e164dvx"
    test_email = "nhce.amit@gmail.com"
    test_rag_id = "68b721bc660680eb1bfbfe4b"
    test_api_key = "sk-default-AP2dc2OE8ElziiXisioG75rOg6EZZ8B8"
    
    print(f"🔧 Test Parameters:")
    print(f"   - User ID: {test_user_id}")
    print(f"   - Email: {test_email}")
    print(f"   - RAG ID: {test_rag_id}")
    print(f"   - Session ID: {test_user_id}+{test_email}")
    
    try:
        print(f"\n🚀 Starting Complete Interview Processing...")
        
        # This would call the actual API to get chat history
        # For testing, you might want to mock this part
        result = process_completed_interview(
            user_id=test_user_id,
            email=test_email,
            rag_id=test_rag_id,
            api_key=test_api_key
        )
        
        print(f"✅ Complete Workflow Completed Successfully!")
        print(f"📊 Processing Result:")
        print(json.dumps(result, indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ Complete Workflow Failed: {e}")
        print(f"Error Type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    print("🧪 Text Training Test Suite")
    print("=" * 60)
    
    # Test 1: Direct text training with sample data
    success1 = test_text_training_with_sample_data()
    
    # Test 2: Complete workflow (requires actual chat history API)
    print(f"\n📋 Note: The complete workflow test requires actual chat history from Lyzr API")
    print(f"   You can uncomment the line below to test with real data:")
    print(f"   # success2 = test_full_workflow()")
    
    # Uncomment this line to test the complete workflow:
    # success2 = test_full_workflow()
    
    print(f"\n🏁 Test Summary:")
    print(f"   - Text Training Test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    # print(f"   - Complete Workflow Test: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Run this test to verify text training works")
    print(f"   2. Test with your actual chat history data")
    print(f"   3. Verify the KB training in Lyzr console")
    print(f"   4. Check that PDFs are uploaded to S3")
