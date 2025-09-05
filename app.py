from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import csv
import io
import os
import time
import json
import requests
import asyncio
import urllib.parse
from pymongo import MongoClient
from bson import ObjectId

from dotenv import load_dotenv
from prompts import INTERVIEW_AGENT_PROMPT, CHAT_AGENT_PROMPT
from pdf_utils import (
    create_simple_pdf_from_text,
    get_chat_history,
    create_lyzr_agent,
    create_lyzr_rag_kb,
    link_agent_with_rag,
    upload_pdf_to_s3,
    train_text_directly,
    generate_presigned_url,
    categorize_chat_history
)
load_dotenv()

URL = os.getenv("MONGODB_URL")
S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION")

app = FastAPI()

# -----------------------------
# MIDDLEWARE
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# -----------------------------
# MONGO CONFIG
# -----------------------------
try:
    client = MongoClient(URL)
    db = client.data_collection_agent
    accounts_col = db.accounts
    emails_col = db.emails
    smtp_col = db.smtp_credentials
    scheduler_col = db.schedulers
    interviews_col = db.interviews
    email_contents_col = db.email_contents
    chat_sessions_col = db.chat_sessions
    categorized_pdfs_col = db.categorized_pdfs
    
    # Test connection
    client.admin.command('ping')
    print("Successfully connected to MongoDB")
    
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    raise

# -----------------------------
# MODELS
# -----------------------------
class Account(BaseModel):
    user_id: str
    api_key: str
    agent_id: Optional[str] = None  # Interview agent ID
    rag_id: Optional[str] = None    # Knowledge base ID
    chat_agent_id: Optional[str] = None  # Chat agent ID (linked with KB)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class Email(BaseModel):
    user_id: str
    email: EmailStr
    status: str = "pending"
    follow_up_count: int = 0
    last_sent_at: Optional[datetime] = None
    error_message: Optional[str] = None  # Add error message field
    created_at: datetime
    updated_at: datetime

class SMTPCreds(BaseModel):
    user_id: str
    username: str
    password: str
    host: str  # SMTP server hostname (e.g., smtp.gmail.com)
    created_at: datetime

class Scheduler(BaseModel):
    user_id: str
    max_limit: int
    interval: int
    time: str
    created_at: datetime

class Interview(BaseModel):
    user_id: str
    email: EmailStr
    token: str
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    
class EmailContent(BaseModel):
    user_id: str
    email: str
    subject: str
    content: str
    follow_up_number: int
    email_status: str = "sent"  # sent, failed, delivered
    created_at: datetime 

class ChatMessage(BaseModel):
    user_id: str
    agent_id: str
    session_id: str
    message: str

class ChatSession(BaseModel):
    user_id: str
    email: EmailStr
    agent_id: Optional[str] = None
    rag_id: Optional[str] = None
    session_status: str = "active"  # active, completed, abandoned
    started_at: datetime
    completed_at: Optional[datetime] = None
    message_count: int = 0
    pdf_s3_url: Optional[str] = None

class InterviewProcessRequest(BaseModel):
    user_id: str
    email: EmailStr

class AgentCreateRequest(BaseModel):
    user_id: str
    name: str = "Interview Agent"
    description: str = "AI agent for conducting interviews"
    token: str  # Lyzr API token from frontend

class ChatAgentCreateRequest(BaseModel):
    user_id: str
    name: str = "Chat Agent"
    description: str = "AI chat agent with knowledge base access"
    token: str  # Lyzr API token from frontend

class AccountUpdateRequest(BaseModel):
    agent_id: Optional[str] = None
    rag_id: Optional[str] = None
    chat_agent_id: Optional[str] = None 

class PDFCategory(BaseModel):
    title: str
    subheading: str
    category: str

class CategorizedPDF(BaseModel):
    user_id: str
    email: EmailStr
    session_id: str
    pdf_s3_url: str
    title: str
    subheading: str
    category: str
    type: str  # "chat_session" or "interview"
    message_count: int = 0
    kb_trained: bool = False
    created_at: datetime
    updated_at: datetime 


# -----------------------------
# ENDPOINTS
# -----------------------------

## Accounts
@app.post("/accounts")
def create_account(account: Account):
    try:
        # Check if account already exists
        existing = accounts_col.find_one({"user_id": account.user_id})
        if existing:
            return {"message": "Account already exists", "user_id": account.user_id}
        
        # Set created_at and updated_at timestamps
        account_data = account.dict()
        account_data["created_at"] = datetime.now(timezone.utc)
        account_data["updated_at"] = datetime.now(timezone.utc)
        
        accounts_col.insert_one(account_data)
        print(f"Account created: {account.user_id}")
        return account_data
        
    except Exception as e:
        print(f"Failed to create account {account.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create account")

@app.get("/accounts/{user_id}")
def get_account(user_id: str):
    try:
        account = accounts_col.find_one({"user_id": user_id})
        if not account:
            raise HTTPException(status_code=404, detail="Account not found")
        
        account["_id"] = str(account["_id"])
        return account
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Failed to get account {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get account")

@app.patch("/accounts/{user_id}")
def update_account(user_id: str, updates: AccountUpdateRequest):
    try:
        update_data = {k: v for k, v in updates.dict().items() if v is not None}
        if not update_data:
            raise HTTPException(status_code=400, detail="No valid updates provided")
            
        update_data["updated_at"] = datetime.now(timezone.utc)
        
        result = accounts_col.update_one(
            {"user_id": user_id},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Account not found")
            
        print(f"Account updated: {user_id}")
        return {"message": "Account updated successfully", "user_id": user_id}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Failed to update account {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update account")

## SMTP Credentials
@app.post("/smtp")
def set_smtp(smtp: SMTPCreds):
    try:
        print(f"Setting SMTP credentials for user: {smtp.user_id}, host: {smtp.host}")
        
        smtp_col.replace_one({"user_id": smtp.user_id}, smtp.dict(), upsert=True)
        
        print(f"Successfully configured SMTP for user: {smtp.user_id}")
        return smtp
        
    except Exception as e:
        print(f"Failed to set SMTP credentials for user {smtp.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to configure SMTP")

@app.get("/smtp/{user_id}")
def get_smtp(user_id: str):
    try:
        print(f"Fetching SMTP credentials for user: {user_id}")
        
        smtp = smtp_col.find_one({"user_id": user_id})
        if not smtp:
            print(f"SMTP config not found for user: {user_id}")
            raise HTTPException(status_code=404, detail="SMTP config not found")
            
        smtp["_id"] = str(smtp["_id"])  # Convert ObjectId to string
        
        return smtp
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Failed to fetch SMTP config for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch SMTP config")

## ------------------------
## AGENT CREATION ENDPOINTS
## ------------------------
@app.post("/agents/create")
def create_agent_with_kb(request: AgentCreateRequest):
    """Create an interview agent and knowledge base (WITHOUT linking them)"""
    try:
        print(f"Creating interview agent and KB for user: {request.user_id}")
        
        # Use the predefined interview agent system prompt
        system_prompt = INTERVIEW_AGENT_PROMPT

        # 1. Create the interview agent
        print(f"Step 1: Creating interview agent with name: {request.name}")
        agent_response = create_lyzr_agent(
            name=request.name,
            system_prompt=system_prompt,
            description=request.description,
            api_key=request.token
        )
        agent_id = agent_response.get("agent_id")
        if not agent_id:
            print(f"Agent creation failed - no ID in response: {agent_response}")
            raise HTTPException(status_code=500, detail=f"Failed to get agent ID from response: {agent_response}")
        
        print(f"Step 1 completed: Interview agent created with ID: {agent_id}")
        
        # 2. Create the knowledge base
        print(f"Step 2: Creating knowledge base")
        kb_response = create_lyzr_rag_kb(
            name=f"{request.name} Knowledge Base",
            api_key=request.token
        )
        rag_id = kb_response.get("id")
        if not rag_id:
            print(f"KB creation failed - no ID in response: {kb_response}")
            raise HTTPException(status_code=500, detail=f"Failed to get knowledge base ID from response: {kb_response}")
        
        print(f"Step 2 completed: Knowledge base created with ID: {rag_id}")
        
        # 3. Update user account with agent and KB IDs (NO LINKING)
        print(f"Step 3: Updating user account with interview agent and KB IDs")
        update_data = {
            "agent_id": agent_id,
            "rag_id": rag_id,
            "updated_at": datetime.now(timezone.utc)
        }
        
        result = accounts_col.update_one(
            {"user_id": request.user_id},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            print(f"Account not found for user {request.user_id}, but agent/KB created successfully")
        else:
            print(f"Step 3 completed: User account updated successfully")
        
        print(f"Interview agent and KB creation completed successfully for user: {request.user_id}")
        return {
            "message": "Interview agent and knowledge base created successfully (not linked)",
            "agent_id": agent_id,
            "rag_id": rag_id,
            "agent_response": agent_response,
            "kb_response": kb_response,
            "note": "Interview agent and KB are created but not linked. Use chat agent creation to link with KB."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Failed to create interview agent and KB for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create interview agent and knowledge base: {str(e)}")

@app.post("/agents/chat/create")
def create_chat_agent_with_kb_link(request: ChatAgentCreateRequest):
    """Create a chat agent and link it with existing knowledge base"""
    try:
        print(f"Creating chat agent for user: {request.user_id}")
        
        # Get user account to fetch existing rag_id
        user_account = accounts_col.find_one({"user_id": request.user_id})
        if not user_account:
            raise HTTPException(status_code=404, detail="User account not found")
        
        rag_id = user_account.get("rag_id")
        if not rag_id:
            raise HTTPException(status_code=400, detail="No knowledge base found. Please create an interview agent first.")
        
        # Use the predefined chat agent system prompt
        system_prompt = CHAT_AGENT_PROMPT

        # 1. Create the chat agent
        print(f"Step 1: Creating chat agent with name: {request.name}")
        agent_response = create_lyzr_agent(
            name=request.name,
            system_prompt=system_prompt,
            description=request.description,
            api_key=request.token
        )
        chat_agent_id = agent_response.get("agent_id")
        if not chat_agent_id:
            print(f"Chat agent creation failed - no ID in response: {agent_response}")
            raise HTTPException(status_code=500, detail=f"Failed to get chat agent ID from response: {agent_response}")
        
        print(f"Step 1 completed: Chat agent created with ID: {chat_agent_id}")
        
        # 2. Link chat agent with existing knowledge base
        print(f"Step 2: Linking chat agent {chat_agent_id} with existing KB {rag_id}")
        link_response = link_agent_with_rag(
            agent_id=chat_agent_id,
            rag_id=rag_id,
            agent_name=request.name,
            agent_system_prompt=system_prompt,
            rag_name=f"{request.name} Knowledge Base",
            api_key=request.token
        )
        
        print(f"Step 2 completed: Chat agent linked with KB successfully")
        
        # 3. Update user account with chat agent ID
        print(f"Step 3: Updating user account with chat agent ID")
        update_data = {
            "chat_agent_id": chat_agent_id,
            "updated_at": datetime.now(timezone.utc)
        }
        
        result = accounts_col.update_one(
            {"user_id": request.user_id},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            print(f"Account not found for user {request.user_id}, but chat agent created successfully")
        else:
            print(f"Step 3 completed: User account updated successfully")
        
        print(f"Chat agent creation and linking completed successfully for user: {request.user_id}")
        return {
            "message": "Chat agent created and linked with knowledge base successfully",
            "chat_agent_id": chat_agent_id,
            "rag_id": rag_id,
            "agent_response": agent_response,
            "link_response": link_response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Failed to create chat agent for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create chat agent: {str(e)}")

@app.get("/agents/{agent_id}/info")
def get_agent_info(agent_id: str):
    """Get public agent information by agent_id"""
    try:
        print(f"Fetching agent info for: {agent_id}")
        
        # Find user account with this agent_id
        user_account = accounts_col.find_one({"agent_id": agent_id})
        if not user_account:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {
            "agent_id": agent_id,
            "user_id": user_account.get("user_id"),
            "agent_prompt": user_account.get("agent_prompt"),
            "rag_id": user_account.get("rag_id"),
            "has_api_key": bool(user_account.get("api_key")),
            "created_at": user_account.get("created_at"),
            "updated_at": user_account.get("updated_at")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Failed to get agent info {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get agent info")


## ----------------
## EMAILS ENDPOINTS
## ----------------
@app.post("/emails")
def add_email(email: Email):
    try:
        emails_col.insert_one(email.dict())
        return email
        
    except Exception as e:
        print(f"Failed to add email {email.email}: {e}")
        raise HTTPException(status_code=500, detail="Failed to add email")

@app.post("/emails/upload-csv")
def upload_emails_csv(user_id: str, file: UploadFile = File(...)):
    try:
        content = file.file.read().decode("utf-8")
        reader = csv.reader(io.StringIO(content))
        header = next(reader)
        
        if "email" not in [h.lower() for h in header]:
            raise HTTPException(status_code=400, detail="CSV must contain an 'email' column")
            
        email_index = [h.lower() for h in header].index("email")

        inserted = 0
        skipped = 0
        
        for row_num, row in enumerate(reader, start=2):  # start=2 because header is row 1
            if len(row) > email_index:
                email_address = row[email_index].strip()
                if email_address:  # Skip empty emails
                    # Check if email already exists for this user
                    existing = emails_col.find_one({"user_id": user_id, "email": email_address})
                    if not existing:
                        email_record = Email(
                            user_id=user_id,
                            email=email_address,
                            created_at=datetime.utcnow(),
                            updated_at=datetime.utcnow()
                        )
                        emails_col.insert_one(email_record.dict())
                        inserted += 1
                    else:
                        skipped += 1
                        
        print(f"CSV uploaded for {user_id}: {inserted} emails added, {skipped} skipped")
        return {"message": f"{inserted} emails added for user {user_id}", "skipped": skipped}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"CSV upload failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to process CSV file")

@app.get("/emails/{user_id}")
def list_emails(user_id: str):
    try:
        records = list(emails_col.find({"user_id": user_id}))
        for record in records:
            record["_id"] = str(record["_id"])
        return records
        
    except Exception as e:
        print(f"Failed to fetch emails for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch emails")

@app.delete("/emails/{user_id}/{email}")
def delete_email_cascade(user_id: str, email: str):
    """Delete an email and all related data (cascade delete)"""
    
    print(f"Starting cascade delete for email {email}, user: {user_id}")
    deleted_counts = {}
    
    try:
        # 1. Delete from emails collection
        email_result = emails_col.delete_many({"user_id": user_id, "email": email})
        deleted_counts["emails"] = email_result.deleted_count
        
        # 2. Delete from interviews collection  
        interview_result = interviews_col.delete_many({"user_id": user_id, "email": email})
        deleted_counts["interviews"] = interview_result.deleted_count
        
        # 3. Delete from email_contents collection (email threads/history)
        content_result = email_contents_col.delete_many({"user_id": user_id, "email": email})
        deleted_counts["email_contents"] = content_result.deleted_count
        
        # Check if anything was actually deleted
        total_deleted = sum(deleted_counts.values())
        
        if total_deleted == 0:
            raise HTTPException(status_code=404, detail=f"No records found for email {email}")
        
        print(f"Successfully completed cascade delete for email {email}, user: {user_id}, total deleted: {total_deleted}")
        
        return {
            "message": f"Successfully deleted email {email} and all related data",
            "email": email,
            "user_id": user_id,
            "deleted_counts": deleted_counts,
            "total_records_deleted": total_deleted
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during cascade delete for email {email}, user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error during cascade delete: {str(e)}")

@app.put("/emails/{user_id}/{email}/status")
def update_email_status(user_id: str, email: str, status: str, error_message: str = None):
    """Update email status (sent, failed, delivered, bounced, etc.)"""
    
    print(f"Updating email status for {email}, user: {user_id}, new status: {status}")
    
    try:
        # Find the email record
        email_record = emails_col.find_one({"user_id": user_id, "email": email})
        if not email_record:
            print(f"Email {email} not found for user {user_id} during status update")
            raise HTTPException(status_code=404, detail=f"Email {email} not found for user {user_id}")
        
        # Prepare update operations
        update_operations = {}
        
        # Set fields to update
        set_fields = {
            "status": status,
            "updated_at": datetime.utcnow()
        }
        
        # Add error message if provided, otherwise unset it for successful statuses
        if error_message:
            set_fields["error_message"] = error_message
            update_operations["$set"] = set_fields
            print(f"Email status update with error for {email}: {error_message}")
        else:
            # For successful statuses, remove error message if it exists
            if status in ["sent", "delivered", "opened", "clicked"]:
                update_operations["$set"] = set_fields
                update_operations["$unset"] = {"error_message": ""}
            else:
                update_operations["$set"] = set_fields
        
        # Update the email record
        result = emails_col.update_one(
            {"user_id": user_id, "email": email},
            update_operations
        )
        
        if result.modified_count == 0:
            print(f"Failed to update email status for {email}, user: {user_id}")
            raise HTTPException(status_code=500, detail="Failed to update email status")
        
        # Return updated record
        updated_record = emails_col.find_one({"user_id": user_id, "email": email})
        updated_record["_id"] = str(updated_record["_id"])
        
        return {
            "message": f"Email status updated to {status}",
            "email": email,
            "status": status,
            "updated_record": updated_record
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating email status for {email}, user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating email status: {str(e)}")

@app.patch("/emails/{user_id}/{email}")
def patch_email_fields(user_id: str, email: str, updates: dict):
    try:
        # Find the email record
        email_record = emails_col.find_one({"user_id": user_id, "email": email})
        if not email_record:
            raise HTTPException(status_code=404, detail=f"Email {email} not found for user {user_id}")
        
        # Allowed fields to update
        allowed_fields = ["status", "follow_up_count", "last_sent_at", "error_message"]
        
        # Filter updates to only allowed fields
        filtered_updates = {k: v for k, v in updates.items() if k in allowed_fields}
        
        if not filtered_updates:
            raise HTTPException(status_code=400, detail="No valid fields provided for update")
        
        # Add updated_at timestamp
        filtered_updates["updated_at"] = datetime.utcnow()
        
        # Update the email record
        result = emails_col.update_one(
            {"user_id": user_id, "email": email},
            {"$set": filtered_updates}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=500, detail="Failed to update email")
        
        # Return updated record
        updated_record = emails_col.find_one({"user_id": user_id, "email": email})
        updated_record["_id"] = str(updated_record["_id"])
        
        return {
            "message": f"Email updated successfully",
            "email": email,
            "updated_fields": filtered_updates,
            "updated_record": updated_record
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating email: {str(e)}")

@app.post("/scheduler")
def set_scheduler(s: Scheduler):
    try:
        print(f"Setting scheduler for user: {s.user_id}, interval: {s.interval}min, time: {s.time}")
        
        scheduler_col.replace_one({"user_id": s.user_id}, s.dict(), upsert=True)
        
        print(f"Successfully configured scheduler for user: {s.user_id}")
        return s
        
    except Exception as e:
        print(f"Failed to set scheduler for user {s.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to configure scheduler")

@app.get("/scheduler/{user_id}")
def get_scheduler(user_id: str):
    try:
        print(f"Fetching scheduler for user: {user_id}")
        
        sched = scheduler_col.find_one({"user_id": user_id})
        if not sched:
            print(f"Scheduler not found for user: {user_id}")
            raise HTTPException(status_code=404, detail="Scheduler not found")
            
        sched["_id"] = str(sched["_id"])
        
        return sched
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Failed to fetch scheduler for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch scheduler")

@app.post("/interview/start")
def start_interview(user_id: str, email: EmailStr):
    try:
        print(f"Starting interview for user: {user_id}, email: {email}")
        
        token = f"{user_id}-{email}"
        interview = Interview(
            user_id=user_id,
            email=email,
            token=token,
            status="started",
            started_at=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
        interviews_col.insert_one(interview.dict())
        
        print(f"Successfully started interview for user: {user_id}, email: {email}, token: {token}")
        
        return interview
        
    except Exception as e:
        print(f"Failed to start interview for user {user_id}, email {email}: {e}")
        raise HTTPException(status_code=500, detail="Failed to start interview")

@app.post("/interview/complete/{token}")
def complete_interview(token: str):
    try:
        print(f"Completing interview with token: {token}")
        
        result = interviews_col.find_one({"token": token})
        if not result:
            print(f"Interview not found for token: {token}")
            raise HTTPException(status_code=404, detail="Interview not found")
        
        user_id = result["user_id"]
        email = result["email"]
        session_id = f"{user_id}+{email}"
        
        # Update interview status
        interviews_col.update_one(
            {"token": token},
            {"$set": {"status": "completed", "completed_at": datetime.utcnow()}}
        )
        
        # Update chat session status if exists
        chat_sessions_col.update_one(
            {"session_id": session_id},
            {"$set": {"session_status": "completed", "completed_at": datetime.utcnow()}}
        )
        
        # Get updated interview
        interview = interviews_col.find_one({"token": token})
        interview["_id"] = str(interview["_id"])
        
        # Add chat session info
        chat_session = chat_sessions_col.find_one({"session_id": session_id})
        if chat_session:
            chat_session["_id"] = str(chat_session["_id"])
            interview["chat_session"] = chat_session
        
        # Add redirect info with agent_id
        user_account = accounts_col.find_one({"user_id": interview["user_id"]})
        agent_id = user_account.get("agent_id") if user_account else None
        
        # URL encode session_id for path parameter
        encoded_session_id = urllib.parse.quote(session_id, safe='')
        
        if agent_id:
            interview["chat_link"] = f"http://localhost:3000/chat/{encoded_session_id}?agent_id={agent_id}"
        else:
            interview["chat_link"] = f"http://localhost:3000/chat/{encoded_session_id}"
            
        interview["should_redirect_to_chat"] = True
        interview["completion_message"] = "Interview completed! You can continue the conversation in the chat interface."
        
        print(f"Successfully completed interview for token: {token}")
        
        return interview
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Failed to complete interview for token {token}: {e}")
        raise HTTPException(status_code=500, detail="Failed to complete interview")

@app.get("/interview/status/{user_id}/{email}")
def check_interview_status(user_id: str, email: EmailStr):
    try:
        print(f"Checking interview status for user: {user_id}, email: {email}")
        
        token = f"{user_id}-{email}"
        result = interviews_col.find_one({"token": token})
        
        status = result.get("status", "pending") if result else "not started"
        print(f"Interview status for {email}: {status}")
        
        return {"status": status}
        
    except Exception as e:
        print(f"Failed to check interview status for user {user_id}, email {email}: {e}")
        raise HTTPException(status_code=500, detail="Failed to check interview status")

@app.get("/interview/{token}")
def get_interview(token: str):
    try:
        print(f"Fetching interview details for token: {token}")
        
        interview = interviews_col.find_one({"token": token})
        if not interview:
            print(f"Interview not found for token: {token}")
            raise HTTPException(status_code=404, detail="Interview not found")
            
        interview["_id"] = str(interview["_id"])
        
        # Add chat session information
        user_id = interview["user_id"]
        email = interview["email"]
        session_id = f"{user_id}+{email}"
        
        # Get chat session if exists
        chat_session = chat_sessions_col.find_one({"session_id": session_id})
        if chat_session:
            chat_session["_id"] = str(chat_session["_id"])
            interview["chat_session"] = chat_session
            
        # Add chat link with agent_id
        user_account = accounts_col.find_one({"user_id": user_id})
        agent_id = user_account.get("agent_id") if user_account else None
        
        # URL encode session_id for path parameter
        encoded_session_id = urllib.parse.quote(session_id, safe='')
        
        if agent_id:
            interview["chat_link"] = f"http://localhost:3000/chat/{encoded_session_id}?agent_id={agent_id}"
        else:
            interview["chat_link"] = f"http://localhost:3000/chat/{encoded_session_id}"
        
        return interview
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Failed to fetch interview for token {token}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch interview")

@app.get("/interview/redirect/{token}")
def redirect_to_chat(token: str):
    """Redirect from old interview URL to new chat URL"""
    try:
        print(f"Redirecting token {token} to chat interface")
        
        interview = interviews_col.find_one({"token": token})
        if not interview:
            raise HTTPException(status_code=404, detail="Interview not found")
        
        user_id = interview["user_id"]
        email = interview["email"]
        session_id = f"{user_id}+{email}"
        
        # Create chat session if it doesn't exist
        session_data = {
            "user_id": user_id,
            "email": email,
            "session_id": session_id,
            "session_status": "active",
            "started_at": datetime.utcnow(),
            "message_count": 0
        }
        
        chat_sessions_col.update_one(
            {"session_id": session_id},
            {"$setOnInsert": session_data},
            upsert=True
        )
        
        chat_url = f"http://localhost:3000/chat/{session_id}"
        
        return {
            "redirect_url": chat_url,
            "session_id": session_id,
            "message": "Please use the chat interface for your interview"
        }
        
    except Exception as e:
        print(f"Failed to redirect token {token}: {e}")
        raise HTTPException(status_code=500, detail="Failed to redirect to chat")

@app.get("/email-thread/{user_id}/{email}")
def get_email_thread(user_id: str, email: str):
    """Get email thread/conversation history for a specific email address"""
    try:
        print(f"Fetching email thread for user: {user_id}, email: {email}")
        
        thread = list(email_contents_col.find(
            {"user_id": user_id, "email": email}
        ).sort("follow_up_number", 1))
        
        if not thread:
            print(f"No email thread found for user: {user_id}, email: {email}")
            return {"email": email, "thread": [], "total_emails": 0}
        
        # Convert ObjectId to string
        for email_record in thread:
            email_record["_id"] = str(email_record["_id"])
        
        return {
            "email": email,
            "thread": thread,
            "total_emails": len(thread),
            "last_sent": thread[-1]["created_at"] if thread else None
        }
        
    except Exception as e:
        print(f"Failed to fetch email thread for user {user_id}, email {email}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch email thread")

@app.get("/email-threads/{user_id}")
def get_all_email_threads(user_id: str):
    """Get all email threads for a user with summary"""
    try:
        print(f"Fetching all email threads for user: {user_id}")
        
        # Get all unique emails for this user
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": "$email",
                "total_emails": {"$sum": 1},
                "last_sent": {"$max": "$created_at"},
                "first_sent": {"$min": "$created_at"},
                "latest_follow_up": {"$max": "$follow_up_number"}
            }},
            {"$sort": {"last_sent": -1}}
        ]
        
        threads_summary = list(email_contents_col.aggregate(pipeline))
        
        result = []
        for thread in threads_summary:
            result.append({
                "email": thread["_id"],
                "total_emails_sent": thread["total_emails"],
                "last_sent": thread["last_sent"],
                "first_sent": thread["first_sent"],
                "latest_follow_up_number": thread["latest_follow_up"]
            })
        
        print(f"Successfully fetched {len(result)} email threads for user: {user_id}")
        
        return {"threads": result, "total_conversations": len(result)}
        
    except Exception as e:
        print(f"Failed to fetch email threads for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch email threads")

## Interview Agent Chat (Streaming, with message count)
@app.post("/chat/interview/send/stream")
async def send_interview_message_stream(chat_msg: ChatMessage):
    """Send message to interview agent (streaming, tracks message count)"""
    try:
        print(f"Sending interview message for user: {chat_msg.user_id}, session: {chat_msg.session_id}")
        
        # Get user account to fetch API key
        user_account = accounts_col.find_one({"user_id": chat_msg.user_id})
        if not user_account:
            # For public access, try to find any user with this agent_id
            user_account = accounts_col.find_one({"agent_id": chat_msg.agent_id})
            if not user_account:
                raise HTTPException(status_code=404, detail="No user found with this agent configuration")
        
        api_key = user_account.get("api_key")
        if not api_key:
            raise HTTPException(status_code=400, detail="No API key configured for this agent")
        
        # Use agent_id from account (interview agent)
        actual_agent_id = user_account.get("agent_id") or chat_msg.agent_id
        if not actual_agent_id:
            raise HTTPException(status_code=400, detail="No interview agent found. Please create an interview agent first.")
        
        # Update or create chat session record WITH message count for interviews
        session_filter = {
            "user_id": chat_msg.user_id,
            "session_id": chat_msg.session_id
        }
        
        # Check if session exists first
        existing_session = chat_sessions_col.find_one(session_filter)
        
        if not existing_session:
            # Create new session
            session_data = {
                "user_id": chat_msg.user_id,
                "session_id": chat_msg.session_id,
                "agent_id": actual_agent_id,
                "session_status": "active",
                "last_message_at": datetime.now(timezone.utc),
                "agent_type": "interview_agent",
                "started_at": datetime.now(timezone.utc),
                "message_count": 1
            }
            chat_sessions_col.insert_one(session_data)
            message_count = 1
        else:
            # Update existing session and increment message count
            session_update = {
                "$set": {
                    "agent_id": actual_agent_id,
                    "session_status": "active",
                    "last_message_at": datetime.now(timezone.utc),
                    "agent_type": "interview_agent"
                },
                "$inc": {
                    "message_count": 1
                }
            }
            
            chat_sessions_col.update_one(session_filter, session_update)
            
            # Get updated message count
            updated_session = chat_sessions_col.find_one(session_filter)
            message_count = updated_session.get("message_count", 1)        # Use streaming response for interview agent (same as chat agent but with message count)
        async def generate_stream():
            try:
                # Send initial message count
                yield f"data: {json.dumps({'message_count': message_count, 'type': 'metadata'})}\n\n"
                
                # Prepare request to Lyzr streaming API
                headers = {
                    'Content-Type': 'application/json',
                    'x-api-key': api_key
                }

                # email = ''
                # if '+' in chat_msg.session_id:
                #     email = chat_msg.session_id.split('+', 1)[1]

                data = {
                    'user_id': chat_msg.user_id,
                    'agent_id': actual_agent_id,
                    'session_id': chat_msg.session_id,
                    'message': chat_msg.message,
                    # 'system_prompt_variables': {
                    #     'session_id': chat_msg.session_id,
                    #     'user_id': chat_msg.user_id,
                    #     'email': email
                    # },
                }
                
                # Make streaming request to Lyzr API
                response = requests.post(
                    "https://agent-prod.studio.lyzr.ai/v3/inference/stream/",
                    headers=headers,
                    json=data,
                    stream=True
                )
                response.raise_for_status()
                
                # Stream the response back to client
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            data_content = line_text[6:]  # Remove 'data: ' prefix
                            if data_content.strip() == '[DONE]':
                                # Send completion signal with final message count
                                yield f"data: {json.dumps({'message_count': message_count, 'type': 'complete'})}\n\n"
                                yield f"data: [DONE]\n\n"
                                break
                            yield f"data: {data_content}\n\n"
                            await asyncio.sleep(0.01)  # Small delay for smooth streaming
                
                # Ensure we always send [DONE] at the end
                yield f"data: [DONE]\n\n"
                        
            except Exception as e:
                print(f"Interview streaming error: {e}")
                yield f"data: {{\"error\": \"Streaming failed\"}}\n\n"
                yield f"data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Failed to send interview message for user {chat_msg.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to send interview message")

## Chat Agent (Streaming, no message count)
@app.post("/chat/agent/send/stream")
async def send_chat_agent_message_stream(chat_msg: ChatMessage):
    """Send message to chat agent with streaming response (no message count tracking)"""
    try:
        print(f"Sending chat agent message for user: {chat_msg.user_id}, session: {chat_msg.session_id}")
        
        # Get user account to fetch API key and chat_agent_id
        user_account = accounts_col.find_one({"user_id": chat_msg.user_id})
        if not user_account:
            raise HTTPException(status_code=404, detail="User account not found")
        
        api_key = user_account.get("api_key")
        if not api_key:
            raise HTTPException(status_code=400, detail="No API key configured for this user")
        
        # Use chat_agent_id from user account (this is the correct agent for chat)
        actual_agent_id = user_account.get("chat_agent_id")
        if not actual_agent_id:
            # Fallback to regular agent_id if chat_agent_id is not set
            actual_agent_id = user_account.get("agent_id") or chat_msg.agent_id
            if not actual_agent_id:
                raise HTTPException(status_code=400, detail="No chat agent found. Please create a chat agent first.")
        
        print(f"🔍 AGENT ID DEBUG:")
        print(f"   - user_account.chat_agent_id: {user_account.get('chat_agent_id')}")
        print(f"   - user_account.agent_id: {user_account.get('agent_id')}")
        print(f"   - chat_msg.agent_id: {chat_msg.agent_id}")
        print(f"   - actual_agent_id selected: {actual_agent_id}")
        
        # Use streaming response for chat agent
        async def generate_stream():
            try:
                headers = {
                    'Content-Type': 'application/json',
                    'x-api-key': api_key
                }
                print(f"🔄 LYZR API REQUEST DEBUG:")
                print(f"   - user_id: {chat_msg.user_id}")
                print(f"   - agent_id: {actual_agent_id}")
                print(f"   - session_id: {chat_msg.session_id}")
                print(f"   - message: {chat_msg.message[:50]}...")
                print(f"   - api_key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}")
                
                data = {
                    'user_id': chat_msg.user_id,
                    'agent_id': actual_agent_id,
                    'session_id': chat_msg.session_id,
                    'message': chat_msg.message,
                    'system_prompt_variables': {}
                }
                
                # Make streaming request to Lyzr API
                response = requests.post(
                    "https://agent-prod.studio.lyzr.ai/v3/inference/stream/",
                    headers=headers,
                    json=data,
                    stream=True
                )
                response.raise_for_status()
                
                # Stream the response back to client
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            data_content = line_text[6:]  # Remove 'data: ' prefix
                            if data_content.strip() == '[DONE]':
                                yield f"data: [DONE]\n\n"
                                break
                            yield f"data: {data_content}\n\n"
                            await asyncio.sleep(0.01)  # Small delay for smooth streaming
                        
            except Exception as e:
                print(f"Chat agent streaming error: {e}")
                yield f"data: {{\"error\": \"Streaming failed\"}}\n\n"
                yield f"data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Failed to send chat agent message for user {chat_msg.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to send chat agent message")


## -------------------
## CHAT HISTORY ENDPOINTS
## -------------------
@app.get("/chat/history/{session_id}")
def get_chat_history_api(session_id: str, agent_id: str = None):
    """Get chat history for a session"""
    try:
        print(f"Fetching chat history for session: {session_id}")
        
        # Parse session_id to get user_id (handle cases where user_id might contain '+')
        # Split on the last '+' to properly separate user_id and email
        last_plus_index = session_id.rfind('+')
        if last_plus_index == -1:
            raise HTTPException(status_code=400, detail="No '+' found in session_id")
        user_id = session_id[:last_plus_index]
        
        # Get user account to fetch API key
        user_account = accounts_col.find_one({"user_id": user_id})
        
        # If no user account found and agent_id provided, try to find by agent_id
        if not user_account and agent_id:
            user_account = accounts_col.find_one({"agent_id": agent_id})
        
        # For backward compatibility, try without API key first
        api_key = user_account.get("api_key") if user_account else None
        
        # Get chat history from Lyzr API
        history = get_chat_history(session_id, api_key)
        
        # Parse session_id to get email for session lookup
        email = session_id[last_plus_index + 1:]
        
        # Try to get session information from database
        session_info = chat_sessions_col.find_one({
            "user_id": user_id,
            "email": email
        })
        
        # If no session exists and no chat history, this is a new user
        if not session_info and not history:
            print(f"New user detected for session {session_id} - no existing session or chat history")
        
        return {
            "session_id": session_id,
            "messages": history if history else [],
            "total_messages": len(history) if history else 0,
            "session": session_info if session_info else None,
            "message_count": session_info.get("message_count", 0) if session_info else 0
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as is
        raise
    except Exception as e:
        print(f"Failed to fetch chat history for session {session_id}: {e}")
        
        # For new users, don't throw an error - return empty response
        if "404" in str(e) or "No messages found" in str(e):
            print(f"No chat history found for new user session {session_id} - returning empty response")
            return {
                "session_id": session_id,
                "messages": [],
                "total_messages": 0,
                "session": None,
                "message_count": 0
            }
        
        raise HTTPException(status_code=500, detail="Failed to fetch chat history")


## -------------------
## INTERVIEW COMPLETE ENDPOINT (PROCESS -> TRAIN -> UPLOAD -> COMPLETE)
## -------------------
@app.post("/chat/interview/complete/{session_id}")
def complete_interview_by_session(session_id: str):
    """Complete interview by session_id - Public endpoint for interview participants"""
    try:
        print(f"Completing interview by session: {session_id}")
        
        # Parse session_id to get user_id and email
        try:
            last_plus_index = session_id.rfind('+')
            if last_plus_index == -1:
                raise ValueError("Invalid session_id format")
            user_id = session_id[:last_plus_index]
            email = session_id[last_plus_index + 1:]
        except:
            print(f"Invalid session_id format: {session_id}")
            raise HTTPException(status_code=400, detail="Invalid session_id format")
        
        # Find or create interview record
        token = f"{user_id}-{email}"
        interview = interviews_col.find_one({"token": token})
        
        if not interview:
            # Create interview record if it doesn't exist
            interview = {
                "user_id": user_id,
                "email": email,
                "token": token,
                "status": "completed",
                "created_at": datetime.utcnow(),
                "completed_at": datetime.utcnow()
            }
            interviews_col.insert_one(interview)
            print(f"Created new interview record for session: {session_id}")
        else:
            # Update existing interview
            interviews_col.update_one(
                {"token": token},
                {"$set": {"status": "completed", "completed_at": datetime.utcnow()}}
            )
            print(f"Updated existing interview record for session: {session_id}")
        
        # Create or update chat session
        chat_session = chat_sessions_col.find_one({"session_id": session_id})
        
        if not chat_session:
            # Create new chat session
            user_account = accounts_col.find_one({"user_id": user_id})
            agent_id = user_account.get("agent_id") if user_account else None
            rag_id = user_account.get("rag_id") if user_account else None
            
            chat_session = {
                "user_id": user_id,
                "email": email,
                "session_id": session_id,
                "agent_id": agent_id,
                "rag_id": rag_id,
                "session_status": "completed",
                "started_at": datetime.utcnow(),
                "completed_at": datetime.utcnow(),
                "message_count": 0
            }
            chat_sessions_col.insert_one(chat_session)
            print(f"Created new chat session for session: {session_id}")
        else:
            # Update existing session
            chat_sessions_col.update_one(
                {"session_id": session_id},
                {"$set": {"session_status": "completed", "completed_at": datetime.utcnow()}}
            )
            print(f"Updated existing chat session for session: {session_id}")
        
        return {
            "message": "Interview completed successfully",
            "session_id": session_id,
            "user_id": user_id,
            "email": email,
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Failed to complete interview for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to complete interview")

@app.post("/interview/process")
def process_interview_completion(request: InterviewProcessRequest):
    """Process completed interview: get chat history, categorize, generate PDF, and upload to S3 (NO KB training)"""
    print("=" * 80)
    print("STARTING INTERVIEW PROCESSING WORKFLOW")
    print("=" * 80)
    
    try:
        token = f"{request.user_id}-{request.email}"
        session_id = f"{request.user_id}+{request.email}"
        
        print(f"🔧 INITIALIZATION:")
        print(f"   - User ID: {request.user_id}")
        print(f"   - Email: {request.email}")
        print(f"   - Generated Token: {token}")
        print(f"   - Generated Session ID: {session_id}")
        
        # Get user account
        print(f"\n🔍 STEP 0: RETRIEVING USER ACCOUNT")
        print(f"   - Searching for user_id: {request.user_id}")
        user_account = accounts_col.find_one({"user_id": request.user_id})
        if not user_account:
            print(f"   ❌ ERROR: User account not found for user_id: {request.user_id}")
            raise HTTPException(status_code=404, detail="User account not found")
        
        print(f"   ✅ User account found")
        print(f"   - Account keys: {list(user_account.keys())}")
        
        api_key = user_account.get("api_key")
        if not api_key:
            print(f"   ❌ ERROR: No API key found in user account")
            raise HTTPException(status_code=400, detail="No API key found for user")
        
        print(f"   ✅ API key found: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}")
        
        # Step 1: Get chat history
        print(f"\n📜 STEP 1: RETRIEVING CHAT HISTORY")
        print(f"   - Session ID: {session_id}")
        print(f"   - API Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}")
        print(f"   - Calling get_chat_history function...")
        
        try:
            chat_history = get_chat_history(session_id=session_id, api_key=api_key)
            print(f"   - get_chat_history returned: {type(chat_history)}")
            print(f"   - Chat history length: {len(chat_history) if chat_history else 0}")
            
            if not chat_history:
                print(f"   ⚠️  WARNING: No chat history found for session: {session_id}")
                print(f"   - This might be a new user with no messages yet")
                print(f"   - Returning empty response instead of error")
                return {
                    "message": "No chat history available for processing",
                    "session_id": session_id,
                    "user_id": request.user_id,
                    "email": request.email,
                    "chat_history": [],
                    "processing_skipped": True
                }
            
            print(f"   ✅ Chat history retrieved successfully")
            print(f"   - Total messages: {len(chat_history)}")
            print(f"   - First few messages preview:")
            for i, msg in enumerate(chat_history[:3]):
                print(f"     [{i}] Type: {type(msg)}, Keys: {list(msg.keys()) if isinstance(msg, dict) else 'Not a dict'}")
                if isinstance(msg, dict) and 'message' in msg:
                    preview = msg['message'][:100] + "..." if len(msg['message']) > 100 else msg['message']
                    print(f"         Message preview: {preview}")
                    
        except Exception as history_error:
            print(f"   ❌ ERROR getting chat history: {history_error}")
            print(f"   ❌ Error type: {type(history_error).__name__}")
            print(f"   ❌ Error details: {str(history_error)}")
            raise
        
        # Step 2: Categorize chat history using AI agent
        print(f"\n🤖 STEP 2: CATEGORIZING CHAT HISTORY")
        print(f"   - Session ID: {session_id}")
        print(f"   - Chat history messages count: {len(chat_history)}")
        
        categorization = {"title": "Conversation Summary", "subheading": "Processed conversation content", "category": "General Discussion"}
        print(f"   - Default categorization prepared: {categorization}")
        
        try:
            print(f"   - Calling categorize_chat_history function...")
            categorization_result = categorize_chat_history(chat_history)
            print(f"   - categorize_chat_history returned: {type(categorization_result)}")
            print(f"   - Categorization result: {categorization_result}")
            
            if categorization_result and isinstance(categorization_result, dict):
                categorization = categorization_result
                print(f"   ✅ Categorization successful")
                print(f"   - Title: {categorization.get('title', 'N/A')}")
                print(f"   - Subheading: {categorization.get('subheading', 'N/A')}")
                print(f"   - Category: {categorization.get('category', 'N/A')}")
            else:
                print(f"   ⚠️ WARNING: Invalid categorization result, using default")
                
        except Exception as cat_error:
            print(f"   ❌ ERROR during categorization: {cat_error}")
            print(f"   ❌ Error type: {type(cat_error).__name__}")
            print(f"   ❌ Error details: {str(cat_error)}")
            print(f"   - Using default categorization: {categorization}")
        
        # Step 3: Generate PDF from chat history
        print(f"\n📄 STEP 3: GENERATING PDF")
        print(f"   - Session ID: {session_id}")
        print(f"   - Converting chat history to JSON...")
        
        try:
            chat_history_json = json.dumps(chat_history, indent=2)
            print(f"   - JSON conversion successful, length: {len(chat_history_json)} characters")
            print(f"   - Calling create_simple_pdf_from_text function...")
            
            pdf_file = create_simple_pdf_from_text(chat_history_json)
            pdf_content = pdf_file.getvalue()
            
            print(f"   ✅ PDF generation successful")
            print(f"   - PDF content size: {len(pdf_content)} bytes")
            print(f"   - PDF file type: {type(pdf_file)}")
            
        except Exception as pdf_error:
            print(f"   ❌ ERROR generating PDF: {pdf_error}")
            print(f"   ❌ Error type: {type(pdf_error).__name__}")
            print(f"   ❌ Error details: {str(pdf_error)}")
            raise
        
        # Step 4: Upload PDF to S3
        print(f"\n☁️ STEP 4: UPLOADING PDF TO S3")
        print(f"   - Session ID: {session_id}")
        print(f"   - User ID: {request.user_id}")
        print(f"   - Email: {request.email}")
        print(f"   - PDF size: {len(pdf_content)} bytes")
        
        s3_upload_success = False
        s3_url = None
        s3_error = None
        signed_url = None
        
        try:
            print(f"   - Calling upload_pdf_to_s3 function...")
            upload_result = upload_pdf_to_s3(pdf_content, request.user_id, request.email, session_id)
            print(f"   - upload_pdf_to_s3 returned: {type(upload_result)}")
            print(f"   - Upload result: {upload_result}")
            
            if isinstance(upload_result, dict):
                s3_url = upload_result.get('s3_url')
                signed_url = upload_result.get('signed_url')
                s3_upload_success = True
                print(f"   ✅ S3 upload successful")
                print(f"   - S3 URL: {s3_url}")
                print(f"   - Signed URL available: {'Yes' if signed_url else 'No'}")
            else:
                print(f"   ⚠️ WARNING: Unexpected upload result format: {upload_result}")
                
        except Exception as s3_exception:
            s3_error = str(s3_exception)
            print(f"   ❌ ERROR during S3 upload: {s3_error}")
            print(f"   ❌ Error type: {type(s3_exception).__name__}")
            print(f"   ❌ Error details: {str(s3_exception)}")
        
        # Step 5: Store categorized PDF information
        print(f"\n💾 STEP 5: STORING CATEGORIZED PDF INFORMATION")
        if s3_upload_success and s3_url:
            print(f"   - S3 upload was successful, proceeding with database storage")
            print(f"   - Session ID: {session_id}")
            
            categorized_pdf_data = {
                "user_id": request.user_id,
                "email": request.email,
                "session_id": session_id,
                "pdf_s3_url": s3_url,
                "title": categorization["title"],
                "subheading": categorization["subheading"], 
                "category": categorization["category"],
                "type": "interview",
                "message_count": len(chat_history),
                "kb_trained": False,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            print(f"   - Categorized PDF data prepared:")
            for key, value in categorized_pdf_data.items():
                print(f"     - {key}: {value}")
            
            try:
                # Store in categorized PDFs collection
                result = categorized_pdfs_col.insert_one(categorized_pdf_data)
                print(f"   ✅ Categorized PDF stored successfully")
                print(f"   - Inserted document ID: {result.inserted_id}")
                print(f"   - Title: {categorization['title']}")
                print(f"   - Category: {categorization['category']}")
            except Exception as db_error:
                print(f"   ❌ ERROR storing categorized PDF: {db_error}")
                print(f"   ❌ Error type: {type(db_error).__name__}")
                raise
        else:
            print(f"   ⚠️ SKIPPING: S3 upload failed, not storing categorized PDF info")
            print(f"   - S3 upload success: {s3_upload_success}")
            print(f"   - S3 URL: {s3_url}")
        
        # Step 6: Update database records
        print(f"\n🗄️ STEP 6: UPDATING DATABASE RECORDS")
        print(f"   - Session ID: {session_id}")
        print(f"   - Token: {token}")
        
        update_data = {
            "status": "completed",
            "processed_at": datetime.utcnow(),
            "session_id": session_id,
            "pdf_generated": True,
            "pdf_size_bytes": len(pdf_content),
            "s3_upload_success": s3_upload_success,
            "chat_messages_count": len(chat_history),
            "categorization": categorization
        }
        
        print(f"   - Update data prepared:")
        for key, value in update_data.items():
            print(f"     - {key}: {value}")
        
        if s3_url:
            update_data["pdf_s3_url"] = s3_url
            print(f"   - Added S3 URL to update data: {s3_url}")
        if s3_error:
            update_data["s3_error"] = s3_error
            print(f"   - Added S3 error to update data: {s3_error}")
        
        # Update interview record
        print(f"   - Updating interview record with token: {token}")
        try:
            interview_result = interviews_col.update_one(
                {"token": token},
                {"$set": update_data},
                upsert=True
            )
            print(f"   ✅ Interview record updated successfully")
            print(f"   - Matched count: {interview_result.matched_count}")
            print(f"   - Modified count: {interview_result.modified_count}")
            print(f"   - Upserted ID: {interview_result.upserted_id}")
        except Exception as interview_db_error:
            print(f"   ❌ ERROR updating interview record: {interview_db_error}")
            raise
        
        # Update chat session record
        print(f"   - Updating chat session record with session_id: {session_id}")
        try:
            session_result = chat_sessions_col.update_one(
                {"session_id": session_id},
                {"$set": {
                    "session_status": "completed",
                    "processed_at": datetime.utcnow(),
                    "pdf_generated": True,
                    "pdf_size_bytes": len(pdf_content),
                    "s3_upload_success": s3_upload_success,
                    "pdf_s3_url": s3_url if s3_url else None,
                    "categorization": categorization
                }}
            )
            print(f"   ✅ Chat session record updated successfully")
            print(f"   - Matched count: {session_result.matched_count}")
            print(f"   - Modified count: {session_result.modified_count}")
        except Exception as session_db_error:
            print(f"   ❌ ERROR updating chat session record: {session_db_error}")
            raise
        
        # Build success message
        success_parts = ["Chat history collected", "Content categorized", "PDF generated"]
        if s3_upload_success:
            success_parts.append("uploaded to S3")
        
        print(f"\n🎉 WORKFLOW COMPLETED SUCCESSFULLY")
        print(f"   - Session ID: {session_id}")
        print(f"   - Success parts: {success_parts}")
        print("=" * 80)
        
        return {
            "message": f"Interview processed successfully: {', '.join(success_parts)}",
            "user_id": request.user_id,
            "email": request.email,
            "session_id": session_id,
            "interview_token": token,
            "workflow_completed": True,
            "step_1_chat_history": {
                "success": True,
                "messages_count": len(chat_history)
            },
            "step_2_categorization": {
                "success": True,
                "categorization": categorization
            },
            "step_3_pdf_creation": {
                "success": True,
                "pdf_size_bytes": len(pdf_content)
            },
            "step_4_s3_upload": {
                "success": s3_upload_success,
                "s3_url": s3_url,
                "signed_url": signed_url,
                "error": s3_error
            },
            "kb_training_note": "KB training is handled separately via dedicated endpoints"
        }
        
    except HTTPException:
        print(f"\n❌ HTTP EXCEPTION OCCURRED")
        print("=" * 80)
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR OCCURRED")
        print(f"   - Error: {e}")
        print(f"   - Error type: {type(e).__name__}")
        print(f"   - Error details: {str(e)}")
        print("=" * 80)
        print(f"Failed to process interview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process interview: {str(e)}")

@app.post("/interview/kb-training")
def train_kb_with_session(request: InterviewProcessRequest):
    """Dedicated endpoint for training KB with chat session data"""
    try:
        session_id = f"{request.user_id}+{request.email}"
        
        # Get user account
        user_account = accounts_col.find_one({"user_id": request.user_id})
        if not user_account:
            raise HTTPException(status_code=404, detail="User account not found")
        
        api_key = user_account.get("api_key")
        rag_id = user_account.get("rag_id")
        
        if not api_key:
            raise HTTPException(status_code=400, detail="No API key found for user")
        if not rag_id:
            raise HTTPException(status_code=400, detail="No RAG ID found for user")
        
        # Get chat history
        print(f"Training KB with session data: {session_id}")
        chat_history = get_chat_history(session_id=session_id, api_key=api_key)
        if not chat_history:
            print(f"No chat history found for session {session_id} - nothing to train")
            return {
                "message": "No chat history available for KB training",
                "session_id": session_id,
                "user_id": request.user_id,
                "email": request.email,
                "training_skipped": True
            }
        
        # Prepare text content for training
        text_content = ""
        for message in chat_history:
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            created_at = message.get('created_at', '')
            
            text_content += f"[{role.upper()}] {content}\n"
            if created_at:
                text_content += f"Time: {created_at}\n"
            text_content += "\n"
        
        # Train KB
        training_success = False
        training_result = {}
        training_error = None
        
        try:
            print(f"Training KB for rag_id: {rag_id}")
            training_result = train_text_directly(
                text_content=text_content,
                rag_id=rag_id,
                api_key=api_key,
                data_parser="simple",
                chunk_size=1000,
                chunk_overlap=100,
                extra_info="{}"
            )
            training_success = True
            print(f"KB training successful for rag_id: {rag_id}")
            
            # Update database records to indicate KB was trained
            chat_sessions_col.update_one(
                {"session_id": session_id},
                {"$set": {
                    "session_status": "completed",
                    "kb_trained": True,
                    "kb_trained_at": datetime.utcnow(),
                    "training_result": training_result
                }}
            )
            
        except Exception as training_exception:
            training_error = str(training_exception)
            training_result = {
                "success": False,
                "error": training_error,
                "error_type": type(training_exception).__name__
            }
            print(f"KB training failed: {training_error}")
            
            # Update database records to indicate training failure
            chat_sessions_col.update_one(
                {"session_id": session_id},
                {"$set": {
                    "session_status": "failed",
                    "kb_trained": False,
                    "kb_training_failed_at": datetime.utcnow(),
                    "training_error": training_error
                }}
            )
        
        return {
            "message": f"KB training {'completed successfully' if training_success else 'failed'}",
            "user_id": request.user_id,
            "email": request.email,
            "session_id": session_id,
            "rag_id": rag_id,
            "training_success": training_success,
            "training_result": training_result,
            "training_error": training_error,
            "chat_messages_processed": len(chat_history),
            "text_content_length": len(text_content)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Failed to train KB with session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to train KB: {str(e)}")


## -------------------
## KNOWLEDGE BASE SIGNED URL ENDPOINTS
## -------------------
def get_signed_pdf_url(pdf_info: dict) -> dict:
    """Generate signed URL for a PDF and return updated PDF info"""
    try:
        if pdf_info.get("pdf_url"):
            signed_url = generate_presigned_url(pdf_info["pdf_url"], 2)  # 2 hours expiration
            if signed_url:
                pdf_info["signed_url"] = signed_url
                pdf_info["signed_url_expires_at"] = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        return pdf_info
    except Exception as e:
        print(f"Failed to generate signed URL for PDF {pdf_info.get('id', 'unknown')}: {e}")
        return pdf_info

@app.get("/knowledge-base/pdfs/{user_id}")
async def get_user_pdfs(user_id: str):
    """Get all PDFs for a user (fallback for non-categorized view)"""
    try:
        print(f"Fetching all PDFs for user: {user_id}")
        
        # Find all categorized PDFs for the user
        categorized_pdfs = list(categorized_pdfs_col.find({
            "user_id": user_id
        }).sort("created_at", -1))
        
        pdfs = []
        for pdf_doc in categorized_pdfs:
            pdf_info = {
                "id": str(pdf_doc.get("_id")),
                "email": pdf_doc.get("email"),
                "session_id": pdf_doc.get("session_id"),
                "pdf_url": pdf_doc.get("pdf_s3_url"),
                "title": pdf_doc.get("title", "Untitled"),
                "subheading": pdf_doc.get("subheading", ""),
                "category": pdf_doc.get("category", "Uncategorized"),
                "completed_at": pdf_doc.get("created_at"),
                "message_count": pdf_doc.get("message_count", 0),
                "type": pdf_doc.get("type", "unknown"),
                "kb_trained": pdf_doc.get("kb_trained", False)
            }
            
            # Generate signed URL for the PDF
            signed_pdf = get_signed_pdf_url(pdf_info)
            pdfs.append(signed_pdf)
        
        print(f"Found {len(pdfs)} PDFs for user {user_id}")
        
        return {
            "user_id": user_id,
            "pdf_count": len(pdfs),
            "pdfs": pdfs
        }
        
    except Exception as e:
        print(f"Failed to fetch PDFs for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch PDFs: {str(e)}")

## -------------------
## CATEGORIZED KNOWLEDGE BASE ENDPOINTS
## -------------------

@app.get("/knowledge-base/categories/{user_id}")
async def get_user_categories(user_id: str):
    """Get all available categories for a user's PDFs"""
    try:
        print(f"Fetching categories for user: {user_id}")
        
        # Get unique categories from categorized PDFs
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": "$category",
                "count": {"$sum": 1},
                "latest_pdf": {"$max": "$created_at"}
            }},
            {"$sort": {"count": -1}}
        ]
        
        categories = list(categorized_pdfs_col.aggregate(pipeline))
        
        result = []
        for cat in categories:
            result.append({
                "category": cat["_id"],
                "count": cat["count"],
                "latest_pdf": cat["latest_pdf"]
            })
        
        print(f"Found {len(result)} categories for user {user_id}")
        
        return {
            "user_id": user_id,
            "categories": result
        }
        
    except Exception as e:
        print(f"Failed to fetch categories for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch categories: {str(e)}")

@app.get("/knowledge-base/pdfs/{user_id}/category/{category}")
async def get_pdfs_by_category(user_id: str, category: str):
    """Get all PDFs for a specific category"""
    try:
        print(f"Fetching PDFs for user: {user_id}, category: {category}")
        
        # Decode URL-encoded category name
        category = urllib.parse.unquote(category)
        
        # Find categorized PDFs
        categorized_pdfs = list(categorized_pdfs_col.find({
            "user_id": user_id,
            "category": category
        }).sort("created_at", -1))
        
        pdfs = []
        for pdf_doc in categorized_pdfs:
            pdf_info = {
                "id": str(pdf_doc.get("_id")),
                "email": pdf_doc.get("email"),
                "session_id": pdf_doc.get("session_id"),
                "pdf_url": pdf_doc.get("pdf_s3_url"),
                "title": pdf_doc.get("title", "Untitled"),
                "subheading": pdf_doc.get("subheading", ""),
                "category": pdf_doc.get("category"),
                "completed_at": pdf_doc.get("created_at"),
                "message_count": pdf_doc.get("message_count", 0),
                "type": pdf_doc.get("type", "unknown"),
                "kb_trained": pdf_doc.get("kb_trained", False)
            }
            
            # Generate signed URL for the PDF
            pdfs.append(get_signed_pdf_url(pdf_info))
        
        print(f"Found {len(pdfs)} PDFs in category '{category}' for user {user_id}")
        
        return {
            "user_id": user_id,
            "category": category,
            "pdf_count": len(pdfs),
            "pdfs": pdfs
        }
        
    except Exception as e:
        print(f"Failed to fetch PDFs for category {category}, user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch PDFs: {str(e)}")

@app.get("/knowledge-base/pdfs-categorized/{user_id}")
async def get_user_categorized_pdfs(user_id: str):
    """Get all categorized PDFs for a user organized by category"""
    try:
        print(f"Fetching categorized PDFs for user: {user_id}")
        
        # Find all categorized PDFs for the user
        categorized_pdfs = list(categorized_pdfs_col.find({
            "user_id": user_id
        }).sort("created_at", -1))
        
        # Group by category
        categories = {}
        total_pdfs = 0
        
        for pdf_doc in categorized_pdfs:
            category = pdf_doc.get("category", "Uncategorized")
            
            if category not in categories:
                categories[category] = []
            
            pdf_info = {
                "id": str(pdf_doc.get("_id")),
                "email": pdf_doc.get("email"),
                "session_id": pdf_doc.get("session_id"),
                "pdf_url": pdf_doc.get("pdf_s3_url"),
                "title": pdf_doc.get("title", "Untitled"),
                "subheading": pdf_doc.get("subheading", ""),
                "category": category,
                "completed_at": pdf_doc.get("created_at"),
                "message_count": pdf_doc.get("message_count", 0),
                "type": pdf_doc.get("type", "unknown"),
                "kb_trained": pdf_doc.get("kb_trained", False)
            }
            
            # Generate signed URL for the PDF
            signed_pdf = get_signed_pdf_url(pdf_info)
            categories[category].append(signed_pdf)
            total_pdfs += 1
        
        print(f"Found {total_pdfs} categorized PDFs in {len(categories)} categories for user {user_id}")
        
        return {
            "user_id": user_id,
            "total_pdfs": total_pdfs,
            "categories": categories
        }
        
    except Exception as e:
        print(f"Failed to fetch categorized PDFs for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch categorized PDFs: {str(e)}")

## -------------------
## PRESIGNED URL ENDPOINT
## -------------------

@app.get("/s3/presigned-url")
def get_presigned_url(s3_url: str, expiration_hours: int = 1):
    """
    Generate a presigned URL for an S3 object
    
    Args:
        s3_url: The S3 URL to generate a presigned URL for
        expiration_hours: Hours until the presigned URL expires (default: 1)
    
    Returns:
        dict: Contains the presigned URL or error message
    """
    try:
        print(f"Generating presigned URL for: {s3_url}")
        print(f"Expiration hours: {expiration_hours}")
        
        # Validate expiration_hours
        if expiration_hours < 1 or expiration_hours > 24:
            raise HTTPException(
                status_code=400, 
                detail="Expiration hours must be between 1 and 24"
            )
        
        # Generate presigned URL
        presigned_url = generate_presigned_url(s3_url, expiration_hours)
        
        if presigned_url is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to generate presigned URL. Please check if the S3 URL is valid."
            )
        
        print(f"Presigned URL generated successfully")
        
        return {
            "presigned_url": presigned_url,
            "original_s3_url": s3_url,
            "expires_in_hours": expiration_hours,
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=expiration_hours)).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Failed to generate presigned URL for {s3_url}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error while generating presigned URL: {str(e)}"
        )