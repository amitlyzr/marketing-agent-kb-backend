#!/usr/bin/env python3
"""
Test script for PDF training workflow
Usage: python test_pdf_training.py <api_key> <rag_id>
"""

import sys
from pdf_utils import pdf_training_workflow

def main():
    if len(sys.argv) != 3:
        print("Usage: python test_pdf_training.py <api_key> <rag_id>")
        print("Example: python test_pdf_training.py your_lyzr_api_key your_rag_id")
        sys.exit(1)
    
    api_key = sys.argv[1]
    rag_id = sys.argv[2]
    
    print(f"Testing PDF training workflow with RAG ID: {rag_id}")
    print(f"API Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}")
    print("-" * 50)
    
    try:
        result = pdf_training_workflow(api_key, rag_id, session_id="mem_cme8pevbm00hi0wmc3e164dvx+nhce.amit@gmail.com")
        
        if result["test_status"] == "success":
            print("✅ PDF Training Test PASSED!")
            print(f"📄 PDF Size: {result['pdf_size']} bytes")
            print(f"🎯 RAG ID: {result['rag_id']}")
            print(f"📝 Text Length: {result['test_text_length']} characters")
            print(f"🚀 Training Response: {result['train_response']}")
        else:
            print("❌ PDF Training Test FAILED!")
            print(f"Error: {result['error']}")
            print(f"Error Type: {result['error_type']}")
            
    except Exception as e:
        print(f"❌ Test script failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
