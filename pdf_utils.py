import os
import io
import json
import boto3
import requests
import tempfile
import time
from datetime import datetime, timezone
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from typing import List, Dict
from logger_config import api_logger, interview_processing_logger
import random
import string
import json
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

LYZR_CHAT_STREAM_API_URL = "https://agent-prod.studio.lyzr.ai/v3/inference/stream/"
LYZR_HISTORY_API_URL = "https://agent-prod.studio.lyzr.ai/v1/sessions/{}/history"
LYZR_TRAIN_PDF_URL = "https://rag-prod.studio.lyzr.ai/v3/train/pdf/"  
LYZR_TRAIN_TXT_URL = "https://rag-prod.studio.lyzr.ai/v3/train/txt/"
LYZR_CREATE_AGENT_URL = "https://agent-prod.studio.lyzr.ai/v3/agents/template/single-task"
LYZR_CREATE_RAG_URL = "https://rag-prod.studio.lyzr.ai/v3/rag/"
LYZR_UPDATE_AGENT_URL = "https://agent-prod.studio.lyzr.ai/v3/agents/template/single-task/{}"

def wait_between_operations(seconds: float = 2.0):
    time.sleep(seconds)

def get_s3_client():
    """Initialize and return S3 client"""
    try:
        api_logger.info("Initializing S3 client")
        
        if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET]):
            missing_vars = []
            if not AWS_ACCESS_KEY_ID:
                missing_vars.append("AWS_ACCESS_KEY_ID")
            if not AWS_SECRET_ACCESS_KEY:
                missing_vars.append("AWS_SECRET_ACCESS_KEY")
            if not S3_BUCKET:
                missing_vars.append("AWS_S3_BUCKET")
            
            error_msg = f"AWS credentials and bucket not properly configured. Missing: {', '.join(missing_vars)}"
            api_logger.error(error_msg)
            raise ValueError(error_msg)
        
        api_logger.info(f"Creating S3 client with region: {AWS_REGION}")
        
        client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        
        api_logger.info("S3 client created successfully")
        return client
        
    except Exception as e:
        api_logger.error(f"Failed to create S3 client: {e}", exc_info=True)
        raise

def create_simple_pdf_from_text(text: str) -> io.BytesIO:
    """
    Create a simple PDF from plain text content
    
    Args:
        text: Text content to convert to PDF
        
    Returns:
        PDF file object (BytesIO)
    """
    try:
        api_logger.info(f"Creating simple PDF from text, length: {len(text)} characters")
        
        buffer = io.BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Create custom style for content
        content_style = ParagraphStyle(
            'Content',
            parent=styles['Normal'],
            fontSize=11,
            textColor='black',
            spaceAfter=12,
            alignment=TA_LEFT,
            leftIndent=0,
            rightIndent=0
        )
        
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=16,
            textColor='darkblue',
            spaceAfter=20,
            alignment=TA_LEFT
        )
        
        # Build story
        story = []
        
        # Add title
        story.append(Paragraph("Training Content", title_style))
        story.append(Spacer(1, 12))
        
        # Add timestamp
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        story.append(Paragraph(f"Generated: {timestamp}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Process text content
        # Handle long text by splitting into paragraphs
        if len(text) > 50000:  # If text is very long, truncate for PDF
            text = text[:50000] + "\n\n[Content truncated for PDF generation]"
        
        # Clean and format text
        text = text.replace('\n', '<br/>')
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        
        # Split into chunks if text is very long to avoid ReportLab issues
        max_chunk_size = 10000
        if len(text) > max_chunk_size:
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            for i, chunk in enumerate(chunks):
                story.append(Paragraph(chunk, content_style))
                if i < len(chunks) - 1:  # Add spacer between chunks except for last
                    story.append(Spacer(1, 12))
        else:
            story.append(Paragraph(text, content_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        api_logger.info(f"Simple PDF created successfully, size: {len(buffer.getvalue())} bytes")
        return buffer
        
    except Exception as e:
        api_logger.error(f"Failed to create simple PDF from text: {e}", exc_info=True)
        raise

def upload_pdf_to_s3(pdf_content: bytes, user_id: str, email: str, session_id: str) -> str:
    """
    Upload PDF to S3 and return the URL
    
    Args:
        pdf_content: PDF content as bytes
        user_id: User ID
        email: Email address
        session_id: Session ID
        
    Returns:
        S3 URL of uploaded PDF
    """
    try:
        interview_processing_logger.info(f"Starting S3 upload for session: {session_id}, user: {user_id}, email: {email}")
        interview_processing_logger.info(f"PDF content size: {len(pdf_content)} bytes")
        
        # Check S3 configuration
        interview_processing_logger.info(f"S3 Configuration - Bucket: {S3_BUCKET}, Region: {AWS_REGION}")
        interview_processing_logger.info(f"AWS credentials available: Access Key: {'Yes' if AWS_ACCESS_KEY_ID else 'No'}, Secret Key: {'Yes' if AWS_SECRET_ACCESS_KEY else 'No'}")
        
        s3_client = get_s3_client()
        interview_processing_logger.info("S3 client initialized successfully")
        
        # Generate S3 key
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        safe_email = email.replace('@', '_at_').replace('.', '_')
        s3_key = f"chat_interviews/{user_id}/{safe_email}/{session_id}_{timestamp}.pdf"
        interview_processing_logger.info(f"Generated S3 key: {s3_key}")
        
        # Upload to S3
        interview_processing_logger.info(f"Starting S3 upload to bucket: {S3_BUCKET}")
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=pdf_content,
            ContentType='application/pdf',
            Metadata={
                'user_id': user_id,
                'email': email,
                'session_id': session_id,
                'generated_at': timestamp
            },
        )
        interview_processing_logger.info(f"S3 upload completed successfully for key: {s3_key}")
        
        # Return S3 URL
        s3_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        interview_processing_logger.info(f"PDF uploaded to S3: {s3_url}")
        return s3_url
        
    except Exception as e:
        interview_processing_logger.error(f"Failed to upload PDF to S3 for session {session_id}: {e}", exc_info=True)
        interview_processing_logger.error(f"Error type: {type(e).__name__}")
        interview_processing_logger.error(f"Error details: {str(e)}")
        raise

def get_chat_history(session_id: str, api_key: str = None) -> List[Dict]:
    """
    Get chat history from Lyzr API
    
    Args:
        session_id: Session ID
        api_key: Lyzr API key (if not provided, uses environment variable)
        
    Returns:
        List of chat messages
    """
    try:
        interview_processing_logger.info(f"Getting chat history for session: {session_id}")
        
        lyzr_key = api_key
        if not lyzr_key:
            interview_processing_logger.error("No Lyzr API key provided for chat history")
            raise ValueError("No Lyzr API key provided")
        
        interview_processing_logger.info(f"Using Lyzr API key: {lyzr_key[:8]}...{lyzr_key[-4:] if len(lyzr_key) > 12 else '***'}")
        
        url = LYZR_HISTORY_API_URL.format(session_id)
        interview_processing_logger.info(f"Chat history URL: {url}")
        
        headers = {
            'accept': 'application/json',
            'x-api-key': lyzr_key
        }
        
        interview_processing_logger.info(f"Request headers (without API key): {dict((k, v) for k, v in headers.items() if k != 'x-api-key')}")
        
        interview_processing_logger.info("Sending chat history request to Lyzr API")
        response = requests.get(url, headers=headers)
        
        interview_processing_logger.info(f"Chat history response status code: {response.status_code}")
        interview_processing_logger.info(f"Chat history response headers: {dict(response.headers)}")
        
        if not response.ok:
            interview_processing_logger.error(f"Chat history request failed with status {response.status_code}")
            interview_processing_logger.error(f"Response content: {response.text}")
            response.raise_for_status()
        
        result = response.json()
        interview_processing_logger.info(f"Chat history response type: {type(result)}")
        
        if isinstance(result, list):
            interview_processing_logger.info(f"Retrieved {len(result)} chat messages")
            if result:
                interview_processing_logger.info(f"Sample message keys: {list(result[0].keys()) if result[0] else 'Empty message'}")
        else:
            interview_processing_logger.info(f"Chat history response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        return result
        
    except Exception as e:
        interview_processing_logger.error(f"Failed to get chat history for session {session_id}: {e}", exc_info=True)
        interview_processing_logger.error(f"Error type: {type(e).__name__}")
        interview_processing_logger.error(f"Error details: {str(e)}")
        raise

def send_chat_message(user_id: str, agent_id: str, session_id: str, message: str, api_key: str = None) -> Dict:
    try:
        lyzr_key = api_key
        if not lyzr_key:
            raise ValueError("No Lyzr API key provided")
            
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': lyzr_key
        }
        data = {
            'user_id': user_id,
            'agent_id': agent_id,
            'session_id': session_id,
            'message': message,
            'system_prompt_variables': {}
        }
        
        # Make streaming request
        response = requests.post(LYZR_CHAT_STREAM_API_URL, headers=headers, json=data, stream=True)
        response.raise_for_status()
        
        # Parse streaming response
        full_response = ""
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    data_content = line_text[6:]  # Remove 'data: ' prefix
                    if data_content.strip() == '[DONE]':
                        break
                    full_response += data_content
        
        return {
            'agent_response': full_response.strip(),
            'session_id': session_id,
            'user_id': user_id
        }
        
    except Exception as e:
        api_logger.error(f"Failed to send chat message: {e}")
        raise

def create_lyzr_agent(name: str, prompt: str, description: str = "AI Interview Agent", api_key: str = None) -> Dict:
    """
    Create a new Lyzr agent
    
    Args:
        name: Agent name
        prompt: Agent system prompt/instructions
        description: Agent description
        api_key: Lyzr API key (if not provided, uses environment variable)
        
    Returns:
        Agent creation response with agent_id
    """
    try:
        lyzr_key = api_key
        if not lyzr_key:
            raise ValueError("No Lyzr API key provided")
            
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'x-api-key': lyzr_key
        }
        
        data = {
            "name": name,
            "description": description,
            "agent_role": "You are an Expert Interview Assistant.",
            "agent_goal": "",
            "agent_instructions": prompt,
            "examples": None,
            "tool": "",
            "tool_usage_description": "{}",
            "provider_id": "OpenAI",
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "top_p": 0.9,
            "llm_credential_id": "lyzr_openai",
            "features": [],
            "managed_agents": [],
            "response_format": {
                "type": "text"
            }
        }
        
        # Log the request data for debugging
        api_logger.info(f"Sending agent creation request: {data}")
        
        response = requests.post(LYZR_CREATE_AGENT_URL, headers=headers, json=data)
        
        # Log response details for debugging
        api_logger.info(f"Agent creation response status: {response.status_code}")
        if not response.ok:
            api_logger.error(f"Agent creation failed with response: {response.text}")
        
        response.raise_for_status()
        
        result = response.json()
        api_logger.info(f"Agent creation response: {result}")
        agent_id = result.get('agent_id')  # Changed from 'id' to 'agent_id'
        if not agent_id:
            api_logger.error(f"No agent ID in response: {result}")
            raise ValueError(f"Agent creation succeeded but no ID returned: {result}")
        
        api_logger.info(f"Agent created successfully with ID: {agent_id}")
        return result
        
    except Exception as e:
        api_logger.error(f"Failed to create Lyzr agent: {e}")
        raise

def create_lyzr_rag_kb(name: str = "Interview Knowledge Base", api_key: str = None) -> Dict:
    try:
        # Use provided api_key or fall back to environment variable
        lyzr_key = api_key
        if not lyzr_key:
            raise ValueError("No Lyzr API key provided")
            
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'x-api-key': lyzr_key
        }
        
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))

        data = {
            "user_id": lyzr_key,
            "name": name,
            "description": "Knowledge base for storing interview conversations and insights",
            "llm_credential_id": "lyzr_openai",
            "embedding_credential_id": "lyzr_openai", 
            "vector_db_credential_id": "lyzr_qdrant",
            "collection_name": f"{name.lower().replace(' ', '_')}_{random_suffix}",
            "llm_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-ada-002",
            "vector_store_provider": "Qdrant [Lyzr]",
            "semantic_data_model": False
        }
        
        # Log the request data for debugging
        api_logger.info(f"Sending RAG KB creation request: {data}")
        
        response = requests.post(LYZR_CREATE_RAG_URL, headers=headers, json=data)
        
        # Log response details for debugging
        api_logger.info(f"RAG KB creation response status: {response.status_code}")
        if not response.ok:
            api_logger.error(f"RAG KB creation failed with response: {response.text}")
        
        response.raise_for_status()
        
        result = response.json()
        api_logger.info(f"RAG KB creation response: {result}")
        rag_id = result.get('id')
        if not rag_id:
            api_logger.error(f"No RAG ID in response: {result}")
            raise ValueError(f"RAG KB creation succeeded but no ID returned: {result}")
        
        api_logger.info(f"RAG KB created successfully with ID: {rag_id}")
        return result
        
    except Exception as e:
        api_logger.error(f"Failed to create Lyzr RAG KB: {e}")
        raise

def link_agent_with_rag(agent_id: str, rag_id: str, agent_name: str, agent_prompt: str, rag_name: str = "Interview Knowledge Base", api_key: str = None) -> Dict:
    """
    Link an agent with a RAG knowledge base by updating the agent configuration
    
    Args:
        agent_id: Agent ID
        rag_id: RAG knowledge base ID
        agent_name: Agent name
        agent_prompt: Agent prompt/instructions
        rag_name: RAG knowledge base name
        api_key: Lyzr API key (if not provided, uses environment variable)
        
    Returns:
        Update operation response
    """
    try:
        # Use provided api_key or fall back to environment variable
        lyzr_key = api_key
        if not lyzr_key:
            raise ValueError("No Lyzr API key provided")
            
        url = LYZR_UPDATE_AGENT_URL.format(agent_id)
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'x-api-key': lyzr_key
        }
        
        data = {
            "name": agent_name,
            "description": f"AI Interview Agent with knowledge base integration",
            "agent_role": "You are an Expert in Knowledge Base Retrieval.",
            "agent_goal": "",
            "agent_instructions": agent_prompt,
            "examples": None,
            "tool": "",
            "tool_usage_description": "{}",
            "provider_id": "Aws-Bedrock",
            "model": "bedrock/amazon.nova-pro-v1:0",
            "temperature": 0.3,
            "top_p": 0.9,
            "llm_credential_id": "lyzr_aws-bedrock",
            "features": [{
                "type": "KNOWLEDGE_BASE",
                "config": {
                    "lyzr_rag": {
                        "base_url": "https://rag-prod.studio.lyzr.ai",
                        "rag_id": rag_id,
                        "rag_name": rag_name,
                        "params": {
                            "top_k": 5,
                            "retrieval_type": "basic",
                            "score_threshold": 0.1
                        }
                    },
                    "agentic_rag": []
                },
                "priority": 0
            }],
            "managed_agents": [],
            "response_format": {"type": "text"}
        }
        
        api_logger.info(f"Sending agent linking request: {data}")
        response = requests.put(url, headers=headers, json=data)
        
        api_logger.info(f"Agent linking response status: {response.status_code}")
        if not response.ok:
            api_logger.error(f"Agent linking failed with response: {response.text}")
        
        response.raise_for_status()
        
        result = response.json()
        api_logger.info(f"Agent {agent_id} linked with RAG {rag_id} successfully")
        return result
        
    except Exception as e:
        api_logger.error(f"Failed to link agent {agent_id} with RAG {rag_id}: {e}")
        raise

def pdf_training_workflow(api_key: str, rag_id: str, session_id) -> Dict:
    """
    Test the complete PDF training workflow with sample content
    
    Args:
        api_key: Lyzr API key
        rag_id: RAG knowledge base ID
        test_text: Optional test text content
        
    Returns:
        Test result
    """
    try:
        api_logger.info("Starting PDF training workflow test")

        test_text = get_chat_history(session_id=session_id, api_key=api_key)
        test_text = json.dumps(test_text)

        api_logger.info("Step 1: Generating PDF from test text")
        pdf_file = create_simple_pdf_from_text(test_text)
        pdf_size = len(pdf_file.getvalue())
        api_logger.info(f"PDF generated successfully, size: {pdf_size} bytes")
        
        api_logger.info("Step 2: Training KB directly with PDF")
        train_response = train_pdf_directly(
            pdf_file=pdf_file,
            rag_id=rag_id,
            api_key=api_key,
            data_parser="llmsherpa",
            chunk_size=1000,
            chunk_overlap=100,
            extra_info="{}"
        )
        
        result = {
            "test_status": "success",
            "pdf_size": pdf_size,
            "rag_id": rag_id,
            "train_response": train_response,
            "test_text_length": len(test_text)
        }
        
        api_logger.info(f"PDF training workflow test completed successfully: {result}")
        return result
        
    except Exception as e:
        api_logger.error(f"PDF training workflow test failed: {e}", exc_info=True)
        result = {
            "test_status": "failed",
            "error": str(e),
            "error_type": type(e).__name__
        }
        return result

def process_completed_interview(user_id: str, email: str, rag_id: str = None, api_key: str = None) -> Dict:
    """
    Process completed interview: get chat history, train KB with text, and upload PDF to S3
    
    UPDATED WORKFLOW ORDER:
    1. Get chat history from Lyzr API
    2. Train KB directly with text content (using text training API)
    3. Generate PDF from chat history
    4. Upload PDF to S3 (for backup/storage)
    
    Args:
        user_id: User ID
        email: Email address
        rag_id: RAG knowledge base ID for training
        api_key: Lyzr API key (if not provided, uses environment variable)
        
    Returns:
        Processing result with KB training, PDF generation and S3 upload status
    """
    try:
        session_id = f"{user_id}+{email}"
        interview_processing_logger.info(f"Starting interview processing for session: {session_id}")
        interview_processing_logger.info(f"Parameters - user_id: {user_id}, email: {email}, rag_id: {rag_id}")
        interview_processing_logger.info(f"API key provided: {'Yes' if api_key else 'No'}")
        
        # Step 1: Get chat history
        interview_processing_logger.info(f"Step 1: Getting chat history for session: {session_id}")
        chat_history = get_chat_history(session_id, api_key)
        
        if not chat_history:
            interview_processing_logger.warning(f"No chat history found for session: {session_id}")
            return {"error": "No chat history found"}
        
        interview_processing_logger.info(f"Retrieved {len(chat_history)} messages from chat history")
        
        # Prepare result object
        result = {
            "session_id": session_id,
            "chat_messages_count": len(chat_history),
            "workflow_order": "1. Text Training -> 2. PDF Generation -> 3. S3 Upload"
        }
        
        # Step 2: Train KB with text content (if rag_id provided)
        if rag_id:
            interview_processing_logger.info(f"Step 2: Training KB with text content for rag_id: {rag_id}")
            
            # Convert chat history to text content
            text_content = ""
            for message in chat_history:
                role = message.get('role', 'unknown')
                content = message.get('content', '')
                created_at = message.get('created_at', '')
                
                text_content += f"[{role.upper()}] {content}\n"
                if created_at:
                    text_content += f"Time: {created_at}\n"
                text_content += "\n"
            
            interview_processing_logger.info(f"Converted chat history to text content, length: {len(text_content)} characters")
            
            try:
                train_response = train_text_directly(
                    text_content=text_content,
                    rag_id=rag_id,
                    api_key=api_key,
                    data_parser="simple",
                    chunk_size=1000,
                    chunk_overlap=100,
                    extra_info="{}"
                )
                
                result.update({
                    "rag_id": rag_id,
                    "kb_training_success": True,
                    "kb_training_response": train_response,
                    "training_content_length": len(text_content)
                })
                interview_processing_logger.info(f"Successfully trained KB with text content for rag_id: {rag_id}")
                
            except Exception as train_error:
                interview_processing_logger.error(f"KB text training failed for rag_id {rag_id}: {train_error}")
                result.update({
                    "rag_id": rag_id,
                    "kb_training_success": False,
                    "kb_training_error": str(train_error),
                    "kb_training_error_type": type(train_error).__name__
                })
        else:
            interview_processing_logger.info("No RAG ID provided, skipping KB training")
            result["kb_training_note"] = "No RAG ID provided"
        
        wait_between_operations(1.0)
        
        # Step 3: Generate PDF
        interview_processing_logger.info(f"Step 3: Generating PDF for session: {session_id}")
        pdf_file = create_simple_pdf_from_text(json.dumps(chat_history, indent=2))
        pdf_content = pdf_file.getvalue()
        result.update({
            "pdf_generated": True,
            "pdf_size_bytes": len(pdf_content)
        })
        interview_processing_logger.info(f"PDF generated successfully, size: {len(pdf_content)} bytes")
        
        wait_between_operations(1.0)
        
        # Step 4: Upload PDF to S3 (for backup/storage)
        try:
            interview_processing_logger.info(f"Step 4: Uploading PDF to S3 for session: {session_id}")
            s3_url = upload_pdf_to_s3(pdf_content, user_id, email, session_id)
            result.update({
                "pdf_s3_url": s3_url,
                "s3_upload_success": True
            })
            interview_processing_logger.info(f"Successfully uploaded PDF to S3 for session: {session_id}")
        except Exception as s3_error:
            interview_processing_logger.warning(f"S3 upload failed for session {session_id}: {s3_error}", exc_info=True)
            result.update({
                "s3_upload_success": False,
                "s3_error": str(s3_error),
                "s3_error_type": type(s3_error).__name__
            })
        
        interview_processing_logger.info(f"Successfully processed interview for session: {session_id}")
        interview_processing_logger.info(f"Final result summary: {dict((k, v) for k, v in result.items() if k not in ['kb_training_response'])}")
        return result
        
    except Exception as e:
        interview_processing_logger.error(f"Failed to process completed interview for session {user_id}+{email}: {e}", exc_info=True)
        interview_processing_logger.error(f"Error type: {type(e).__name__}")
        interview_processing_logger.error(f"Error details: {str(e)}")
        raise

def train_text_directly(
        text_content: str,
        rag_id: str,
        api_key: str = None,
        data_parser: str = "simple",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        extra_info: str = "{}"
    ) -> Dict:
    """
    Train RAG knowledge base directly with text content using Lyzr text training API
    
    Args:
        text_content: Text content to train
        rag_id: RAG knowledge base ID
        api_key: Lyzr API key
        data_parser: Parser to use (simple, llmsherpa, pymupdf, unstructured)
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        extra_info: Additional JSON info
        
    Returns:
        Training response
    """
    try:
        interview_processing_logger.info(f"Starting direct text training for rag_id: {rag_id}")
        interview_processing_logger.info(f"Text content length: {len(text_content)} characters")
        
        lyzr_key = api_key
        if not lyzr_key:
            interview_processing_logger.error("No Lyzr API key provided for direct text training")
            raise ValueError("No Lyzr API key provided")

        url = f"{LYZR_TRAIN_TXT_URL}?rag_id={rag_id}"
        interview_processing_logger.info(f"Training URL: {url}")
        
        interview_processing_logger.info("Preparing multipart/form-data payload for text training")
        
        # Create a text file in memory
        text_bytes = text_content.encode('utf-8')
        text_file_bytes = io.BytesIO(text_bytes)
        
        files = {
            'file': ('chat_interview.txt', text_file_bytes, 'text/plain')
        }
        
        # Form data fields - all as strings for multipart/form-data
        data = {
            'data_parser': data_parser,
            'chunk_size': str(chunk_size),
            'chunk_overlap': str(chunk_overlap),
            'extra_info': extra_info
        }
        
        # Headers - don't set Content-Type, let requests handle multipart/form-data encoding
        headers = {
            'accept': 'application/json',
            'x-api-key': lyzr_key
        }
        
        interview_processing_logger.info(f"File: chat_interview.txt, size: {len(text_bytes)} bytes, type: text/plain")
        interview_processing_logger.info(f"Request headers (without API key): {dict((k, v) for k, v in headers.items() if k != 'x-api-key')}")
        
        interview_processing_logger.info("🚀 Sending direct text training request to Lyzr API with multipart/form-data")
        interview_processing_logger.info(f"📤 Upload Details:")
        interview_processing_logger.info(f"   - URL: {url}")
        interview_processing_logger.info(f"   - Method: POST")
        interview_processing_logger.info(f"   - Content-Type: multipart/form-data (auto-generated)")
        interview_processing_logger.info(f"   - File name: chat_interview.txt")
        interview_processing_logger.info(f"   - File size: {len(text_bytes)} bytes")
        interview_processing_logger.info(f"   - RAG ID: {rag_id}")
        interview_processing_logger.info(f"   - Data Parser: {data_parser}")
        interview_processing_logger.info(f"   - Chunk Size: {chunk_size}")
        interview_processing_logger.info(f"   - Chunk Overlap: {chunk_overlap}")
        
        response = requests.post(url, files=files, data=data, headers=headers, timeout=120)

        interview_processing_logger.info(f"📥 Training response status code: {response.status_code}")
        interview_processing_logger.info(f"📥 Training response headers: {dict(response.headers)}")
        interview_processing_logger.info(f"📥 Training response content: {response.text}")

        if not response.ok:
            interview_processing_logger.error(f"❌ Direct text training failed with status {response.status_code}")
            interview_processing_logger.error(f"❌ Response content: {response.text}")
            interview_processing_logger.error(f"❌ This suggests the upload format or parameters are incorrect")
            response.raise_for_status()
        
        try:
            result = response.json()
            interview_processing_logger.info(f"✅ Direct text training completed successfully for rag_id: {rag_id}")
            interview_processing_logger.info(f"📊 Training response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            interview_processing_logger.info(f"📊 Training result: {result}")
            return result
        except json.JSONDecodeError as json_err:
            interview_processing_logger.error(f"❌ Failed to parse JSON response: {json_err}")
            interview_processing_logger.error(f"❌ Raw response: {response.text}")
            raise Exception(f"Invalid JSON response from training API: {response.text}")
            
    except Exception as e:
        interview_processing_logger.error(f"Failed to train text directly for rag_id {rag_id}: {e}", exc_info=True)
        interview_processing_logger.error(f"Error type: {type(e).__name__}")
        interview_processing_logger.error(f"Error details: {str(e)}")
        raise

def train_pdf_directly(
        pdf_file: io.BytesIO,
        rag_id: str,
        api_key: str = None,
        data_parser: str = "llmsherpa",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        extra_info: str = "{}"
    ) -> Dict:
    """
    Train RAG knowledge base directly with PDF file object (deployment-ready, no temp files)
    
    Args:
        pdf_file: PDF file object (BytesIO)
        rag_id: RAG knowledge base ID
        api_key: Lyzr API key
        data_parser: Parser to use (llmsherpa, pymupdf, unstructured)
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        extra_info: Additional JSON info
        
    Returns:
        Training response
    """
    try:
        # Get PDF content as bytes from the file object
        pdf_file.seek(0)  # Ensure we're at the beginning of the file
        pdf_content = pdf_file.getvalue()
        
        interview_processing_logger.info(f"Starting direct PDF training for rag_id: {rag_id}")
        interview_processing_logger.info(f"PDF content size: {len(pdf_content)} bytes")
        
        lyzr_key = api_key
        if not lyzr_key:
            interview_processing_logger.error("No Lyzr API key provided for direct PDF training")
            raise ValueError("No Lyzr API key provided")

        url = f"{LYZR_TRAIN_PDF_URL}?rag_id={rag_id}"
        interview_processing_logger.info(f"Training URL: {url}")
        
        interview_processing_logger.info("Preparing multipart/form-data payload in memory (deployment-ready)")
        
        # Reset file pointer and use the file object directly
        pdf_file.seek(0)
        files = {
            'file': ('chat_interview.pdf', pdf_file, 'application/pdf')
        }
        
        # Form data fields - all as strings for multipart/form-data
        data = {
            'data_parser': data_parser,
            'chunk_size': str(chunk_size),
            'chunk_overlap': str(chunk_overlap),
            'extra_info': extra_info
        }
        
        # Headers - don't set Content-Type, let requests handle multipart/form-data encoding
        headers = {
            'accept': 'application/json',
            'x-api-key': lyzr_key
        }
        
        interview_processing_logger.info(f"File: chat_interview.pdf, size: {len(pdf_content)} bytes, type: application/pdf")
        interview_processing_logger.info(f"Request headers (without API key): {dict((k, v) for k, v in headers.items() if k != 'x-api-key')}")
        
        interview_processing_logger.info("🚀 Sending direct PDF training request to Lyzr API with multipart/form-data")
        interview_processing_logger.info(f"📤 Upload Details:")
        interview_processing_logger.info(f"   - URL: {url}")
        interview_processing_logger.info(f"   - Method: POST")
        interview_processing_logger.info(f"   - Content-Type: multipart/form-data (auto-generated)")
        interview_processing_logger.info(f"   - File name: chat_interview.pdf")
        interview_processing_logger.info(f"   - File size: {len(pdf_content)} bytes")
        interview_processing_logger.info(f"   - RAG ID: {rag_id}")
        interview_processing_logger.info(f"   - Data Parser: {data_parser}")
        interview_processing_logger.info(f"   - Chunk Size: {chunk_size}")
        interview_processing_logger.info(f"   - Chunk Overlap: {chunk_overlap}")
        
        # Use longer timeout and let requests handle multipart encoding
        interview_processing_logger.info({
            "action": "send_training_request",
            "file" : pdf_file,
            "url": url,
            "method": "POST",
            "headers": {k: v for k, v in headers.items() if k != 'x-api-key'},
            "file_size": len(pdf_file.getvalue()),
            "data": data
        })
        
        response = requests.post(url, files=files, data=data, headers=headers, timeout=120)

        interview_processing_logger.info(f"📥 Training response status code: {response.status_code}")
        interview_processing_logger.info(f"📥 Training response headers: {dict(response.headers)}")
        interview_processing_logger.info(f"📥 Training response content: {response.text}")

        if not response.ok:
            interview_processing_logger.error(f"❌ Direct PDF training failed with status {response.status_code}")
            interview_processing_logger.error(f"❌ Response content: {response.text}")
            interview_processing_logger.error(f"❌ This suggests the upload format or parameters are incorrect")
            response.raise_for_status()
        
        try:
            result = response.json()
            interview_processing_logger.info(f"✅ Direct PDF training completed successfully for rag_id: {rag_id}")
            interview_processing_logger.info(f"📊 Training response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            interview_processing_logger.info(f"📊 Training result: {result}")
            return result
        except json.JSONDecodeError as json_err:
            interview_processing_logger.error(f"❌ Failed to parse JSON response: {json_err}")
            interview_processing_logger.error(f"❌ Raw response: {response.text}")
            raise Exception(f"Invalid JSON response from training API: {response.text}")
            
    except Exception as e:
        interview_processing_logger.error(f"Failed to train PDF directly for rag_id {rag_id}: {e}", exc_info=True)
        interview_processing_logger.error(f"Error type: {type(e).__name__}")
        interview_processing_logger.error(f"Error details: {str(e)}")
        raise
