#!/usr/bin/env python3
"""
Test script for interview completion flow
Usage: python test_interview_flow.py <api_key> <rag_id> <user_id> <email>
"""

import sys
from pdf_utils import process_completed_interview

def main():
    if len(sys.argv) != 5:
        print("Usage: python test_interview_flow.py <api_key> <rag_id> <user_id> <email>")
        print("Example: python test_interview_flow.py your_lyzr_api_key your_rag_id test_user test@example.com")
        sys.exit(1)
    
    api_key = sys.argv[1]
    rag_id = sys.argv[2]
    user_id = sys.argv[3]
    email = sys.argv[4]
    
    print(f"Testing Interview Completion Flow")
    print(f"User ID: {user_id}")
    print(f"Email: {email}")
    print(f"RAG ID: {rag_id}")
    print(f"API Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}")
    print("-" * 60)
    
    try:
        result = process_completed_interview(
            user_id=user_id,
            email=email,
            rag_id=rag_id,
            api_key=api_key
        )
        
        print("📊 INTERVIEW PROCESSING RESULTS:")
        print(f"✅ Session ID: {result.get('session_id', 'N/A')}")
        print(f"💬 Chat Messages: {result.get('chat_messages_count', 'N/A')}")
        print(f"📄 PDF Generated: {result.get('pdf_generated', 'N/A')}")
        print(f"📏 PDF Size: {result.get('pdf_size_bytes', 'N/A')} bytes")
        
        if result.get('kb_trained'):
            print("🎯 KB Training: ✅ SUCCESS")
            print(f"🔗 Training Method: {result.get('training_method', 'N/A')}")
            print(f"📝 Training Response: {result.get('train_response', 'N/A')}")
        else:
            print("❌ KB Training: FAILED")
            print(f"🚨 Error: {result.get('kb_error', 'N/A')}")
            print(f"🔍 Error Type: {result.get('kb_error_type', 'N/A')}")
        
        if result.get('s3_upload_success'):
            print("☁️ S3 Upload: ✅ SUCCESS")
            print(f"🔗 S3 URL: {result.get('pdf_s3_url', 'N/A')}")
        else:
            print("☁️ S3 Upload: ❌ FAILED")
            print(f"🚨 S3 Error: {result.get('s3_error', 'N/A')}")
        
        print("\n" + "="*60)
        if result.get('kb_trained'):
            print("🎉 INTERVIEW FLOW TEST: PASSED!")
        else:
            print("💥 INTERVIEW FLOW TEST: FAILED!")
            
    except Exception as e:
        print(f"❌ Test script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
