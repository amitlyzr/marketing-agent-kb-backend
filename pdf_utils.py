import os
import io
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

# def generate_conversation_pdf(chat_history: List[Dict], user_id: str, email: str, session_id: str) -> bytes:
    """
    Generate a beautifully formatted PDF from chat conversation history
    
    Args:
        chat_history: List of chat messages with role, content, and created_at
        user_id: User ID
        email: Email address
        session_id: Session ID
        
    Returns:
        PDF content as bytes
    """
    try:
        api_logger.info(f"Generating conversation PDF for session: {session_id}")
        
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
        
        # Create custom styles for conversation
        title_style = ParagraphStyle(
            'ConversationTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor='darkblue',
            spaceAfter=30,
            alignment=TA_LEFT
        )
        
        metadata_style = ParagraphStyle(
            'Metadata',
            parent=styles['Normal'],
            fontSize=11,
            textColor='gray',
            spaceAfter=20,
            alignment=TA_LEFT
        )
        
        user_message_style = ParagraphStyle(
            'UserMessage',
            parent=styles['Normal'],
            fontSize=12,
            textColor='darkblue',
            leftIndent=0,
            rightIndent=50,
            spaceAfter=8,
            spaceBefore=8,
            alignment=TA_LEFT,
            borderWidth=1,
            borderColor='lightblue',
            borderPadding=10,
            backColor='aliceblue'
        )
        
        assistant_message_style = ParagraphStyle(
            'AssistantMessage',
            parent=styles['Normal'],
            fontSize=12,
            textColor='darkgreen',
            leftIndent=50,
            rightIndent=0,
            spaceAfter=8,
            spaceBefore=8,
            alignment=TA_LEFT,
            borderWidth=1,
            borderColor='lightgreen',
            borderPadding=10,
            backColor='honeydew'
        )
        
        timestamp_style = ParagraphStyle(
            'Timestamp',
            parent=styles['Normal'],
            fontSize=9,
            textColor='gray',
            spaceAfter=4,
            alignment=TA_RIGHT
        )
        
        separator_style = ParagraphStyle(
            'Separator',
            parent=styles['Normal'],
            fontSize=1,
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Build story
        story = []
        
        # Add title
        story.append(Paragraph("💬 Chat Conversation", title_style))
        
        # Add metadata section
        metadata_lines = [
            f"<b>Participant:</b> {email}",
            f"<b>Session ID:</b> {session_id}",
            f"<b>User ID:</b> {user_id}",
            f"<b>Generated:</b> {datetime.now(timezone.utc).strftime('%B %d, %Y at %I:%M %p UTC')}",
            f"<b>Total Messages:</b> {len(chat_history)}"
        ]
        
        for line in metadata_lines:
            story.append(Paragraph(line, metadata_style))
        
        story.append(Spacer(1, 20))
        
        # Add conversation header
        story.append(Paragraph("<b>Conversation History</b>", styles['Heading2']))
        story.append(Spacer(1, 15))
        
        # Add chat messages
        for i, message in enumerate(chat_history):
            role = message.get('role', 'unknown')
            content = message.get('content', '').strip()
            created_at = message.get('created_at', '')
            
            if not content:
                continue
            
            # Format timestamp
            timestamp_text = ""
            if created_at:
                try:
                    if isinstance(created_at, str):
                        # Parse ISO format timestamp
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        timestamp_text = dt.strftime('%I:%M %p')
                    else:
                        timestamp_text = str(created_at)
                except:
                    timestamp_text = str(created_at)
            
            # Clean and format content
            content = content.replace('\n', '<br/>')
            content = content.replace('<', '&lt;').replace('>', '&gt;')
            
            # Add message based on role
            if role == 'user':
                story.append(Paragraph(f"<b>👤 You</b> {timestamp_text}", timestamp_style))
                story.append(Paragraph(content, user_message_style))
            elif role == 'assistant':
                story.append(Paragraph(f"<b>🤖 Assistant</b> {timestamp_text}", timestamp_style))
                story.append(Paragraph(content, assistant_message_style))
            else:
                # Handle unknown roles
                story.append(Paragraph(f"<b>{role.title()}</b> {timestamp_text}", timestamp_style))
                story.append(Paragraph(content, styles['Normal']))
            
            # Add separator except for last message
            if i < len(chat_history) - 1:
                story.append(Spacer(1, 8))
        
        # Add footer
        story.append(Spacer(1, 30))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            textColor='gray',
            alignment=TA_LEFT
        )
        story.append(Paragraph("End of Conversation", footer_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        pdf_content = buffer.getvalue()
        
        api_logger.info(f"Conversation PDF generated successfully, size: {len(pdf_content)} bytes")
        return pdf_content
        
    except Exception as e:
        api_logger.error(f"Failed to generate conversation PDF: {e}", exc_info=True)
        raise

# def generate_pdf_from_chat_history(chat_history: List[Dict], user_id: str, email: str) -> bytes:
    """
    Generate PDF from chat history
    
    Args:
        chat_history: List of chat messages with role and content
        user_id: User ID
        email: Email address
        
    Returns:
        PDF content as bytes
    """
    interview_processing_logger.info(f"Starting PDF generation for user: {user_id}, email: {email}")
    interview_processing_logger.info(f"Chat history contains {len(chat_history)} messages")
    
    buffer = io.BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='black',
        spaceAfter=30,
        alignment=TA_LEFT
    )
    
    user_style = ParagraphStyle(
        'UserMessage',
        parent=styles['Normal'],
        fontSize=12,
        textColor='blue',
        leftIndent=0,
        spaceAfter=12,
        alignment=TA_LEFT
    )
    
    assistant_style = ParagraphStyle(
        'AssistantMessage',
        parent=styles['Normal'],
        fontSize=12,
        textColor='black',
        leftIndent=20,
        spaceAfter=12,
        alignment=TA_LEFT
    )
    
    timestamp_style = ParagraphStyle(
        'Timestamp',
        parent=styles['Normal'],
        fontSize=10,
        textColor='gray',
        spaceAfter=6,
        alignment=TA_RIGHT
    )
    
    # Build story
    story = []
    
    # Add title
    title = f"Chat Interview Session - {email}"
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 12))
    
    # Add metadata
    metadata = f"User ID: {user_id}<br/>Email: {email}<br/>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    story.append(Paragraph(metadata, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Add chat messages
    for message in chat_history:
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        created_at = message.get('created_at', '')
        
        # Add timestamp if available
        if created_at:
            story.append(Paragraph(f"Time: {created_at}", timestamp_style))
        
        # Add message based on role
        if role == 'user':
            story.append(Paragraph(f"<b>User:</b> {content}", user_style))
        elif role == 'assistant':
            story.append(Paragraph(f"<b>Assistant:</b> {content}", assistant_style))
        else:
            story.append(Paragraph(f"<b>{role.title()}:</b> {content}", styles['Normal']))
        
        story.append(Spacer(1, 6))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    pdf_content = buffer.getvalue()
    
    interview_processing_logger.info(f"PDF generation completed successfully")
    interview_processing_logger.info(f"Generated PDF size: {len(pdf_content)} bytes")
    
    return pdf_content

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

# def process_completed_interview(user_id: str, email: str, rag_id: str = None, api_key: str = None) -> Dict:
    """
    Process completed interview: generate PDF and upload to S3
    
    UPDATED WORKFLOW ORDER:
    1. Get chat history from Lyzr API
    2. Generate PDF from chat history
    3. Upload PDF to S3 (for backup/storage)
    
    Note: KB training is now handled separately by pdf_training_workflow() 
    which should be called BEFORE this function.
    
    Args:
        user_id: User ID
        email: Email address
        rag_id: Optional RAG knowledge base ID (for reference only)
        api_key: Lyzr API key (if not provided, uses environment variable)
        
    Returns:
        Processing result with PDF generation and S3 upload status
    """
    try:
        session_id = f"{user_id}+{email}"
        api_logger.info(f"Starting interview processing for session: {session_id}")
        api_logger.info(f"Parameters - user_id: {user_id}, email: {email}, rag_id: {rag_id}")
        api_logger.info(f"API key provided: {'Yes' if api_key else 'No'}")
        
        # Get chat history
        api_logger.info(f"Step 1: Getting chat history for session: {session_id}")
        chat_history = get_chat_history(session_id, api_key)
        
        if not chat_history:
            api_logger.warning(f"No chat history found for session: {session_id}")
            return {"error": "No chat history found"}
        
        api_logger.info(f"Retrieved {len(chat_history)} messages from chat history")
        
        # Generate PDF
        api_logger.info(f"Step 2: Generating PDF for session: {session_id}")
        pdf_file = create_simple_pdf_from_text(json.dumps(chat_history))
        api_logger.info(f"PDF generated successfully, size: {len(pdf_file.getvalue())} bytes")

        result = {
            "session_id": session_id,
            "chat_messages_count": len(chat_history),
            "pdf_generated": True,
            "pdf_size_bytes": len(pdf_file.getvalue())
        }
        
        wait_between_operations(1.0)
        
        if rag_id:
            api_logger.info(f"RAG ID {rag_id} provided - training should have been handled by pdf_training_workflow()")
            result.update({
                "rag_id": rag_id,
                "kb_training_note": "Training handled separately by pdf_training_workflow"
            })
            
            # Convert PDF bytes to file object for training
            pdf_file = io.BytesIO(pdf_file.getvalue())

            train_response = train_pdf_directly(
                    pdf_file=pdf_file,
                    rag_id=rag_id,
                    api_key=api_key,
                    data_parser="llmsherpa",
                    chunk_size=1000,
                    chunk_overlap=100,
                    extra_info={}
                )

            if train_response.get("success"):
                api_logger.info(f"Successfully trained RAG with ID {rag_id} for session: {session_id}")
            else:
                api_logger.warning(f"RAG training failed for ID {rag_id}: {train_response.get('error')}")
        else:
            api_logger.info("No RAG ID provided, skipping KB training note")
        
        try:
            api_logger.info(f"Step 3: Uploading PDF to S3 for session: {session_id}")
            s3_url = upload_pdf_to_s3(pdf_content, user_id, email, session_id)
            result["pdf_s3_url"] = s3_url
            result["s3_upload_success"] = True
            api_logger.info(f"Successfully uploaded PDF to S3 for session: {session_id}")
        except Exception as s3_error:
            api_logger.warning(f"S3 upload failed for session {session_id}: {s3_error}", exc_info=True)
            result["s3_upload_success"] = False
            result["s3_error"] = str(s3_error)
            result["s3_error_type"] = type(s3_error).__name__
        
        api_logger.info(f"Successfully processed interview for session: {session_id}")
        api_logger.info(f"Final result: {result}")
        return result
        
    except Exception as e:
        api_logger.error(f"Failed to process completed interview for session {user_id}+{email}: {e}", exc_info=True)
        api_logger.error(f"Error type: {type(e).__name__}")
        api_logger.error(f"Error details: {str(e)}")
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
            'extra_info': {}
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
