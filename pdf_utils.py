import os
import io
import json
import boto3
import requests
import time
import random
import json
import string

from urllib.parse import urlparse
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from typing import List, Dict
from datetime import datetime, timedelta
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

LYZR_CHAT_STREAM_API_URL = "https://agent-prod.studio.lyzr.ai/v3/inference/stream/"
LYZR_HISTORY_API_URL = "https://agent-prod.studio.lyzr.ai/v1/sessions/{}/history" 
LYZR_TRAIN_TXT_URL = "https://rag-prod.studio.lyzr.ai/v3/train/txt/"
LYZR_CREATE_AGENT_URL = "https://agent-prod.studio.lyzr.ai/v3/agents/template/single-task"
LYZR_CREATE_RAG_URL = "https://rag-prod.studio.lyzr.ai/v3/rag/"
LYZR_UPDATE_AGENT_URL = "https://agent-prod.studio.lyzr.ai/v3/agents/template/single-task/{}"
LYZR_CHAT_API_URL = "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"

LYZR_ONLY_FOR_CATEGORY_API_KEY = os.getenv("LYZR_API_KEY");
LYZR_CATEGORY_AGENT_ID = os.getenv("LYZR_CATEGORY_AGENT_ID");

def categorize_chat_history(chat_history: List[Dict]) -> Dict:
    """
    Categorize chat history using AI agent to generate title, subheading, and category
    
    Args:
        chat_history: List of chat messages
        agent_id: Agent ID for categorization
        api_key: Lyzr API key
        
    Returns:
        Dict containing title, subheading, and category
    """
    try:
        lyzr_key = LYZR_ONLY_FOR_CATEGORY_API_KEY
        agent_id = LYZR_CATEGORY_AGENT_ID
        
        if not lyzr_key:
            raise ValueError("No Lyzr API key provided")
        
        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
            'x-api-key': lyzr_key
        }
        
        # Generate a unique session ID for this categorization request
        session_id = f"categorization_{int(time.time())}_{random.randint(1000, 9999)}"
        
        data = {
            'user_id': 'default_user',
            'system_prompt_variables': {},
            'agent_id': agent_id,
            'session_id': session_id,
            'message': chat_history
        }
        
        print(f"Sending categorization request to Lyzr API with agent: {agent_id}")
        response = requests.post(LYZR_CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        agent_response = result.get('agent_response', '').strip()
        
        print(f"Categorization response: {agent_response}")
        
        try:
            start_idx = agent_response.find('{')
            end_idx = agent_response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = agent_response[start_idx:end_idx]
                category_data = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['title', 'subheading', 'category']
                if all(field in category_data for field in required_fields):
                    return category_data
                else:
                    print(f"Missing required fields in categorization response: {category_data}")
                    
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from categorization response: {e}")
        
        # Fallback categorization if parsing fails
        print("Using fallback categorization")
        return {
            "title": "Conversation Summary",
            "subheading": "Processed conversation content",
            "category": "General Discussion"
        }
        
    except Exception as e:
        print(f"ERROR: Failed to categorize chat history: {e}")
        # Return default categorization on error
        return {
            "title": "Conversation Summary", 
            "subheading": "Processed conversation content",
            "category": "General Discussion"
        }

def wait_between_operations(seconds: float = 2.0):
    time.sleep(seconds)

def get_s3_client():
    """Initialize and return S3 client"""
    try:
        print("Initializing S3 client")
        
        if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET]):
            missing_vars = []
            if not AWS_ACCESS_KEY_ID:
                missing_vars.append("AWS_ACCESS_KEY_ID")
            if not AWS_SECRET_ACCESS_KEY:
                missing_vars.append("AWS_SECRET_ACCESS_KEY")
            if not S3_BUCKET:
                missing_vars.append("AWS_S3_BUCKET")
            
            error_msg = f"AWS credentials and bucket not properly configured. Missing: {', '.join(missing_vars)}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        print(f"Creating S3 client with region: {AWS_REGION}")
        
        client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        
        print("S3 client created successfully")
        return client
        
    except Exception as e:
        print(f"ERROR: Failed to create S3 client: {e}")
        raise

def create_simple_pdf_from_text(text: str) -> io.BytesIO:
    """
    Create a professionally formatted PDF from chat interview content
    
    Args:
        text: JSON text content containing chat history
        
    Returns:
        PDF file object (BytesIO)
    """
    try:
        print(f"Creating formatted PDF from text, length: {len(text)} characters")
        
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
        
        # Parse JSON content to extract chat data
        try:
            import json
            chat_data = json.loads(text)
            
            # Extract session info
            session_id = chat_data.get('session_id', 'Unknown')
            user_email = chat_data.get('user_email', 'Unknown')
            messages = chat_data.get('messages', [])
            
            # Extract user ID and email from session_id if available
            if '+' in session_id and '@' in session_id:
                parts = session_id.split('+')
                if len(parts) > 1:
                    user_email = parts[1]
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse JSON, treating as plain text: {e}")
            # Fallback to plain text processing
            session_id = "Unknown"
            user_email = "Unknown"
            messages = []
        
        # Create custom styles
        title_style = ParagraphStyle(
            'MainTitle',
            parent=styles['Title'],
            fontSize=18,
            textColor=colors.black,
            spaceAfter=20,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        )
        
        header_style = ParagraphStyle(
            'Header',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.grey,
            spaceAfter=15,
            alignment=TA_LEFT,
            fontName='Helvetica'
        )
        
        user_style = ParagraphStyle(
            'UserMessage',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.blue,
            spaceAfter=8,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold',
            leftIndent=20
        )
        
        assistant_style = ParagraphStyle(
            'AssistantMessage',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.black,
            spaceAfter=12,
            alignment=TA_LEFT,
            fontName='Helvetica',
            leftIndent=20
        )
        
        timestamp_style = ParagraphStyle(
            'Timestamp',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            spaceAfter=5,
            alignment=TA_RIGHT,
            fontName='Helvetica'
        )
        
        # Build story
        story = []
        
        # Add main title
        story.append(Paragraph(f"Chat Interview Session - {user_email}", title_style))
        story.append(Spacer(1, 12))
        
        # Add session info
        story.append(Paragraph(f"User ID: {session_id}", header_style))
        story.append(Paragraph(f"Email: {user_email}", header_style))
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        story.append(Paragraph(f"Generated: {timestamp}", header_style))
        story.append(Spacer(1, 20))
        
        # Process messages if available
        if messages:
            for i, message in enumerate(messages):
                role = message.get('role', 'unknown')
                content = message.get('content', '')
                created_at = message.get('created_at', '')
                
                # Format timestamp
                if created_at:
                    try:
                        # Parse ISO timestamp and format nicely
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Include milliseconds
                        story.append(Paragraph(f"Time: {formatted_time}", timestamp_style))
                    except:
                        story.append(Paragraph(f"Time: {created_at}", timestamp_style))
                
                # Add message content
                if role == 'user':
                    story.append(Paragraph(f"<b>User:</b> {content}", user_style))
                elif role == 'assistant':
                    story.append(Paragraph(f"<b>Assistant:</b> {content}", assistant_style))
                else:
                    story.append(Paragraph(f"<b>{role.title()}:</b> {content}", assistant_style))
                
                # Add spacing between messages
                if i < len(messages) - 1:
                    story.append(Spacer(1, 15))
        else:
            # Fallback: display raw text content in a formatted way
            content_style = ParagraphStyle(
                'Content',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.black,
                spaceAfter=12,
                alignment=TA_LEFT,
                leftIndent=0,
                rightIndent=0,
                fontName='Courier'
            )
            
            # Clean and format text
            clean_text = text.replace('<', '&lt;').replace('>', '&gt;')
            # Split into smaller chunks to avoid ReportLab issues
            max_chunk_size = 5000
            if len(clean_text) > max_chunk_size:
                chunks = [clean_text[i:i+max_chunk_size] for i in range(0, len(clean_text), max_chunk_size)]
                for chunk in chunks:
                    story.append(Paragraph(chunk, content_style))
                    story.append(Spacer(1, 12))
            else:
                story.append(Paragraph(clean_text, content_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        print(f"Formatted PDF created successfully, size: {len(buffer.getvalue())} bytes")
        return buffer
        
    except Exception as e:
        print(f"ERROR: Failed to create simple PDF from text: {e}")
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
        print(f"Starting S3 upload for session: {session_id}, user: {user_id}, email: {email}")
        print(f"PDF content size: {len(pdf_content)} bytes")
        
        # Check S3 configuration
        print(f"S3 Configuration - Bucket: {S3_BUCKET}, Region: {AWS_REGION}")
        print(f"AWS credentials available: Access Key: {'Yes' if AWS_ACCESS_KEY_ID else 'No'}, Secret Key: {'Yes' if AWS_SECRET_ACCESS_KEY else 'No'}")
        
        s3_client = get_s3_client()
        print("S3 client initialized successfully")
        
        # Generate S3 key
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        safe_email = email.replace('@', '_at_').replace('.', '_')
        s3_key = f"chat_interviews/{user_id}/{safe_email}/{session_id}_{timestamp}.pdf"
        print(f"Generated S3 key: {s3_key}")
        
        # Upload to S3
        print(f"Starting S3 upload to bucket: {S3_BUCKET}")
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
        print(f"S3 upload completed successfully for key: {s3_key}")
        
        # Generate S3 URL and signed URL
        s3_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        
        print(f"PDF uploaded to S3: {s3_url}")
        print(f"Generated signed URL (expires in 1 hour)")
        
        return {
            's3_url': s3_url,
            'signed_url_expires_at': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        }
        
    except Exception as e:
        print(f"ERROR: Failed to upload PDF to S3 for session {session_id}: {e}")
        print(f"ERROR: Error type: {type(e).__name__}")
        print(f"ERROR: Error details: {str(e)}")
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
        print(f"Getting chat history for session: {session_id}")
        
        lyzr_key = api_key
        if not lyzr_key:
            print("ERROR: No Lyzr API key provided for chat history")
            raise ValueError("No Lyzr API key provided")
        
        print(f"Using Lyzr API key: {lyzr_key[:8]}...{lyzr_key[-4:] if len(lyzr_key) > 12 else '***'}")
        
        url = LYZR_HISTORY_API_URL.format(session_id)
        print(f"Chat history URL: {url}")
        
        headers = {
            'accept': 'application/json',
            'x-api-key': lyzr_key
        }
        
        print(f"Request headers (without API key): {dict((k, v) for k, v in headers.items() if k != 'x-api-key')}")
        
        print("Sending chat history request to Lyzr API")
        response = requests.get(url, headers=headers)
        
        print(f"Chat history response status code: {response.status_code}")
        print(f"Chat history response headers: {dict(response.headers)}")
        
        if not response.ok:
            print(f"ERROR: Chat history request failed with status {response.status_code}")
            print(f"ERROR: Response content: {response.text}")
            response.raise_for_status()
        
        result = response.json()
        print(f"Chat history response type: {type(result)}")
        
        if isinstance(result, list):
            print(f"Retrieved {len(result)} chat messages")
            if result:
                print(f"Sample message keys: {list(result[0].keys()) if result[0] else 'Empty message'}")
        else:
            print(f"Chat history response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        return result
        
    except Exception as e:
        print(f"ERROR: Failed to get chat history for session {session_id}: {e}")
        print(f"ERROR: Error type: {type(e).__name__}")
        print(f"ERROR: Error details: {str(e)}")
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
        print(f"ERROR: Failed to send chat message: {e}")
        raise

def create_lyzr_agent(name: str, system_prompt: str, description: str = "AI Interview Agent", api_key: str = None) -> Dict:
    """
    Create a new Lyzr agent
    
    Args:
        name: Agent name
        system_prompt: Agent system prompt/instructions
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
            "agent_role": "",
            "agent_goal": "",
            "agent_instructions": system_prompt,
            "examples": None,
            "tool": "",
            "tool_usage_description": "{}",
            "provider_id": "OpenAI",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "top_p": 0.9,
            "llm_credential_id": "lyzr_openai",
            "features": [],
            "managed_agents": [],
            "response_format": {
                "type": "text"
            },
            "store_messages": True
        }
        
        # Log the request data for debugging
        print(f"Sending agent creation request: {data}")
        
        response = requests.post(LYZR_CREATE_AGENT_URL, headers=headers, json=data)
        
        # Log response details for debugging
        print(f"Agent creation response status: {response.status_code}")
        if not response.ok:
            print(f"ERROR: Agent creation failed with response: {response.text}")
        
        response.raise_for_status()
        
        result = response.json()
        print(f"Agent creation response: {result}")
        agent_id = result.get('agent_id')  # Changed from 'id' to 'agent_id'
        if not agent_id:
            print(f"ERROR: No agent ID in response: {result}")
            raise ValueError(f"Agent creation succeeded but no ID returned: {result}")
        
        print(f"Agent created successfully with ID: {agent_id}")
        return result
        
    except Exception as e:
        print(f"ERROR: Failed to create Lyzr agent: {e}")
        raise

def create_lyzr_rag_kb(name: str = "Interview Knowledge Base", api_key: str = None) -> Dict:
    try:
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
        print(f"Sending RAG KB creation request: {data}")
        
        response = requests.post(LYZR_CREATE_RAG_URL, headers=headers, json=data)
        
        # Log response details for debugging
        print(f"RAG KB creation response status: {response.status_code}")
        if not response.ok:
            print(f"ERROR: RAG KB creation failed with response: {response.text}")
        
        response.raise_for_status()
        
        result = response.json()
        print(f"RAG KB creation response: {result}")
        rag_id = result.get('id')
        if not rag_id:
            print(f"ERROR: No RAG ID in response: {result}")
            raise ValueError(f"RAG KB creation succeeded but no ID returned: {result}")
        
        print(f"RAG KB created successfully with ID: {rag_id}")
        return result
        
    except Exception as e:
        print(f"ERROR: Failed to create Lyzr RAG KB: {e}")
        raise

def link_agent_with_rag(agent_id: str, rag_id: str, agent_name: str, agent_system_prompt: str, rag_name: str = "Interview Knowledge Base", api_key: str = None) -> Dict:
    """
    Link an agent with a RAG knowledge base by updating the agent configuration
    
    Args:
        agent_id: Agent ID
        rag_id: RAG knowledge base ID
        agent_name: Agent name
        agent_system_prompt: Agent system prompt/instructions
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
            "description": f"AI Chat Agent with knowledge base integration",
            "agent_role": "",
            "agent_goal": "",
            "agent_instructions": agent_system_prompt,
            "examples": None,
            "tool": "",
            "tool_usage_description": "{}",
            "provider_id": "OpenAI",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "top_p": 0.9,
            "llm_credential_id": "lyzr_openai",
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
            "response_format": {"type": "text"},
            "store_messages": True
        }
        
        print(f"Sending agent linking request: {data}")
        response = requests.put(url, headers=headers, json=data)
        
        print(f"Agent linking response status: {response.status_code}")
        if not response.ok:
            print(f"ERROR: Agent linking failed with response: {response.text}")
        
        response.raise_for_status()
        
        result = response.json()
        print(f"Agent {agent_id} linked with RAG {rag_id} successfully")
        return result
        
    except Exception as e:
        print(f"ERROR: Failed to link agent {agent_id} with RAG {rag_id}: {e}")
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
        print(f"Starting direct text training for rag_id: {rag_id}")
        print(f"Text content length: {len(text_content)} characters")
        
        lyzr_key = api_key
        if not lyzr_key:
            print("ERROR: No Lyzr API key provided for direct text training")
            raise ValueError("No Lyzr API key provided")

        url = f"{LYZR_TRAIN_TXT_URL}?rag_id={rag_id}"
        print(f"Training URL: {url}")
        
        print("Preparing multipart/form-data payload for text training")
        
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
        
        print(f"File: chat_interview.txt, size: {len(text_bytes)} bytes, type: text/plain")
        print(f"Request headers (without API key): {dict((k, v) for k, v in headers.items() if k != 'x-api-key')}")
        
        print("🚀 Sending direct text training request to Lyzr API with multipart/form-data")
        print(f"📤 Upload Details:")
        print(f"   - URL: {url}")
        print(f"   - Method: POST")
        print(f"   - Content-Type: multipart/form-data (auto-generated)")
        print(f"   - File name: chat_interview.txt")
        print(f"   - File size: {len(text_bytes)} bytes")
        print(f"   - RAG ID: {rag_id}")
        print(f"   - Data Parser: {data_parser}")
        print(f"   - Chunk Size: {chunk_size}")
        print(f"   - Chunk Overlap: {chunk_overlap}")
        
        response = requests.post(url, files=files, data=data, headers=headers, timeout=120)

        print(f"📥 Training response status code: {response.status_code}")
        print(f"📥 Training response headers: {dict(response.headers)}")
        print(f"📥 Training response content: {response.text}")

        if not response.ok:
            print(f"❌ ERROR: Direct text training failed with status {response.status_code}")
            print(f"❌ ERROR: Response content: {response.text}")
            print(f"❌ This suggests the upload format or parameters are incorrect")
            response.raise_for_status()
        
        try:
            result = response.json()
            print(f"✅ Direct text training completed successfully for rag_id: {rag_id}")
            print(f"📊 Training response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            print(f"📊 Training result: {result}")
            return result
        except json.JSONDecodeError as json_err:
            print(f"❌ ERROR: Failed to parse JSON response: {json_err}")
            print(f"❌ ERROR: Raw response: {response.text}")
            raise Exception(f"Invalid JSON response from training API: {response.text}")
            
    except Exception as e:
        print(f"ERROR: Failed to train text directly for rag_id {rag_id}: {e}")
        print(f"ERROR: Error type: {type(e).__name__}")
        print(f"ERROR: Error details: {str(e)}")
        raise

def generate_presigned_url(s3_url, expiration_hours=1):
    """
    Generate presigned URL from S3 URL stored in database.
    
    Args:
        s3_url (str): S3 URL from database
        expiration_hours (int): Hours until URL expires
    
    Returns:
        str: Presigned URL or None if error
    """
    try:
        parsed = urlparse(s3_url)
        object_key = parsed.path.lstrip('/')
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': os.getenv('AWS_S3_BUCKET', 'lyzr-marketplace-s3'),
                'Key': object_key
            },
            ExpiresIn=expiration_hours * 3600
        )
        
        return presigned_url
        
    except:
        return None