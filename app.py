from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime, timezone
import csv
import io
import os
import time
import json
import requests
import asyncio
import urllib.parse
from pymongo import MongoClient

from dotenv import load_dotenv
from logger_config import (
    api_logger, 
    log_api_request, 
    log_api_response, 
    log_database_operation,
    log_email_operation
)
from pdf_utils import (
    send_chat_message,
    get_chat_history,
    process_completed_interview,
    create_lyzr_agent,
    create_lyzr_rag_kb,
    link_agent_with_rag,
)
load_dotenv()

URL = os.getenv("MONGODB_URL")

app = FastAPI()

# -----------------------------
# LOGGING MIDDLEWARE
# -----------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    user_id = request.path_params.get('user_id', 'unknown')
    log_api_request(api_logger, request.url.path, request.method, user_id)
    response = await call_next(request)
    execution_time = (time.time() - start_time) * 1000
    log_api_response(api_logger, request.url.path, response.status_code, execution_time, user_id)

    return response

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
    
    # Test connection
    client.admin.command('ping')
    api_logger.info("Successfully connected to MongoDB", extra={'database': 'data_collection_agent'})
    
except Exception as e:
    api_logger.critical(f"Failed to connect to MongoDB: {e}", extra={'error_type': 'database_connection'})
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
    agent_prompt: Optional[str] = None
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
    rag_id: Optional[str] = None 

class AgentCreateRequest(BaseModel):
    user_id: str
    prompt: str
    name: str = "Interview Agent"
    description: str = "AI agent for conducting interviews"
    token: str  # Lyzr API token from frontend

class ChatAgentCreateRequest(BaseModel):
    user_id: str
    prompt: str
    name: str = "Chat Agent"
    description: str = "AI chat agent with knowledge base access"
    token: str  # Lyzr API token from frontend

class AccountUpdateRequest(BaseModel):
    agent_id: Optional[str] = None
    rag_id: Optional[str] = None
    chat_agent_id: Optional[str] = None
    agent_prompt: Optional[str] = None 



# -----------------------------
# ENDPOINTS
# -----------------------------

## Acccount
@app.post("/accounts")
def create_account(account: Account):
    try:
        # Check if account already exists
        existing = accounts_col.find_one({"user_id": account.user_id})
        if existing:
            return {"message": "Account already exists", "user_id": account.user_id}
        
        accounts_col.insert_one(account.dict())
        api_logger.info(f"Account created: {account.user_id}")
        return account
        
    except Exception as e:
        api_logger.error(f"Failed to create account {account.user_id}: {e}")
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
        api_logger.error(f"Failed to get account {user_id}: {e}")
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
            
        api_logger.info(f"Account updated: {user_id}")
        return {"message": "Account updated successfully", "user_id": user_id}
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to update account {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update account")

@app.post("/agents/create")
def create_agent_with_kb(request: AgentCreateRequest):
    """Create an interview agent and knowledge base (WITHOUT linking them)"""
    try:
        api_logger.info(f"Creating interview agent and KB for user: {request.user_id}")
        
        sanitized_prompt = request.prompt.replace('\r', '').replace('\t', ' ').replace('\n', ' ').strip()

        # 1. Create the interview agent
        api_logger.info(f"Step 1: Creating interview agent with name: {request.name}")
        agent_response = create_lyzr_agent(
            name=request.name,
            prompt=sanitized_prompt,
            description=request.description,
            api_key=request.token
        )
        agent_id = agent_response.get("agent_id")
        if not agent_id:
            api_logger.error(f"Agent creation failed - no ID in response: {agent_response}")
            raise HTTPException(status_code=500, detail=f"Failed to get agent ID from response: {agent_response}")
        
        api_logger.info(f"Step 1 completed: Interview agent created with ID: {agent_id}")
        
        # 2. Create the knowledge base
        api_logger.info(f"Step 2: Creating knowledge base")
        kb_response = create_lyzr_rag_kb(
            name=f"{request.name} Knowledge Base",
            api_key=request.token
        )
        rag_id = kb_response.get("id")
        if not rag_id:
            api_logger.error(f"KB creation failed - no ID in response: {kb_response}")
            raise HTTPException(status_code=500, detail=f"Failed to get knowledge base ID from response: {kb_response}")
        
        api_logger.info(f"Step 2 completed: Knowledge base created with ID: {rag_id}")
        
        # 3. Update user account with agent and KB IDs (NO LINKING)
        api_logger.info(f"Step 3: Updating user account with interview agent and KB IDs")
        update_data = {
            "agent_id": agent_id,
            "rag_id": rag_id,
            "agent_prompt": request.prompt,
            "updated_at": datetime.now(timezone.utc)
        }
        
        result = accounts_col.update_one(
            {"user_id": request.user_id},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            api_logger.warning(f"Account not found for user {request.user_id}, but agent/KB created successfully")
        else:
            api_logger.info(f"Step 3 completed: User account updated successfully")
        
        api_logger.info(f"Interview agent and KB creation completed successfully for user: {request.user_id}")
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
        api_logger.error(f"Failed to create interview agent and KB for user {request.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create interview agent and knowledge base: {str(e)}")

@app.post("/agents/chat/create")
def create_chat_agent_with_kb_link(request: ChatAgentCreateRequest):
    """Create a chat agent and link it with existing knowledge base"""
    try:
        api_logger.info(f"Creating chat agent for user: {request.user_id}")
        
        # Get user account to fetch existing rag_id
        user_account = accounts_col.find_one({"user_id": request.user_id})
        if not user_account:
            raise HTTPException(status_code=404, detail="User account not found")
        
        rag_id = user_account.get("rag_id")
        if not rag_id:
            raise HTTPException(status_code=400, detail="No knowledge base found. Please create an interview agent first.")
        
        sanitized_prompt = request.prompt.replace('\r', '').replace('\t', ' ').replace('\n', ' ').strip()

        # 1. Create the chat agent
        api_logger.info(f"Step 1: Creating chat agent with name: {request.name}")
        agent_response = create_lyzr_agent(
            name=request.name,
            prompt=sanitized_prompt,
            description=request.description,
            api_key=request.token
        )
        chat_agent_id = agent_response.get("agent_id")
        if not chat_agent_id:
            api_logger.error(f"Chat agent creation failed - no ID in response: {agent_response}")
            raise HTTPException(status_code=500, detail=f"Failed to get chat agent ID from response: {agent_response}")
        
        api_logger.info(f"Step 1 completed: Chat agent created with ID: {chat_agent_id}")
        
        # 2. Link chat agent with existing knowledge base
        api_logger.info(f"Step 2: Linking chat agent {chat_agent_id} with existing KB {rag_id}")
        link_response = link_agent_with_rag(
            agent_id=chat_agent_id,
            rag_id=rag_id,
            agent_name=request.name,
            agent_prompt=request.prompt,
            rag_name=f"{request.name} Knowledge Base",
            api_key=request.token
        )
        
        api_logger.info(f"Step 2 completed: Chat agent linked with KB successfully")
        
        # 3. Update user account with chat agent ID
        api_logger.info(f"Step 3: Updating user account with chat agent ID")
        update_data = {
            "chat_agent_id": chat_agent_id,
            "updated_at": datetime.now(timezone.utc)
        }
        
        result = accounts_col.update_one(
            {"user_id": request.user_id},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            api_logger.warning(f"Account not found for user {request.user_id}, but chat agent created successfully")
        else:
            api_logger.info(f"Step 3 completed: User account updated successfully")
        
        api_logger.info(f"Chat agent creation and linking completed successfully for user: {request.user_id}")
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
        api_logger.error(f"Failed to create chat agent for user {request.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create chat agent: {str(e)}")

@app.get("/agents/{agent_id}/info")
def get_agent_info(agent_id: str):
    """Get public agent information by agent_id"""
    try:
        api_logger.info(f"Fetching agent info for: {agent_id}")
        
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
        api_logger.error(f"Failed to get agent info {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get agent info")


## Emails Collection
@app.post("/emails")
def add_email(email: Email):
    try:
        emails_col.insert_one(email.dict())
        return email
        
    except Exception as e:
        api_logger.error(f"Failed to add email {email.email}: {e}")
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
                        
        api_logger.info(f"CSV uploaded for {user_id}: {inserted} emails added, {skipped} skipped")
        return {"message": f"{inserted} emails added for user {user_id}", "skipped": skipped}
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"CSV upload failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to process CSV file")

@app.get("/emails/{user_id}")
def list_emails(user_id: str):
    try:
        records = list(emails_col.find({"user_id": user_id}))
        for record in records:
            record["_id"] = str(record["_id"])
        return records
        
    except Exception as e:
        api_logger.error(f"Failed to fetch emails for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch emails")

@app.delete("/emails/{user_id}/{email}")
def delete_email_cascade(user_id: str, email: str):
    """Delete an email and all related data (cascade delete)"""
    
    api_logger.info(f"Starting cascade delete for email {email}, user: {user_id}")
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
            api_logger.warning(f"No records found for cascade delete: {email}, user: {user_id}")
            raise HTTPException(status_code=404, detail=f"No records found for email {email}")
        
        log_database_operation(api_logger, "DELETE_CASCADE", "multiple", user_id, 
                             f"Deleted {total_deleted} records for email: {email}")
        api_logger.info(f"Successfully completed cascade delete for email {email}, user: {user_id}, total deleted: {total_deleted}")
        
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
        api_logger.error(f"Error during cascade delete for email {email}, user {user_id}: {e}",
                        extra={'user_id': user_id, 'email': email, 'error_type': 'cascade_delete'})
        raise HTTPException(status_code=500, detail=f"Error during cascade delete: {str(e)}")

@app.put("/emails/{user_id}/{email}/status")
def update_email_status(user_id: str, email: str, status: str, error_message: str = None):
    """Update email status (sent, failed, delivered, bounced, etc.)"""
    
    api_logger.info(f"Updating email status for {email}, user: {user_id}, new status: {status}")
    
    try:
        # Find the email record
        email_record = emails_col.find_one({"user_id": user_id, "email": email})
        if not email_record:
            api_logger.warning(f"Email {email} not found for user {user_id} during status update")
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
            api_logger.warning(f"Email status update with error for {email}: {error_message}")
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
            api_logger.error(f"Failed to update email status for {email}, user: {user_id}")
            raise HTTPException(status_code=500, detail="Failed to update email status")
        
        # Return updated record
        updated_record = emails_col.find_one({"user_id": user_id, "email": email})
        updated_record["_id"] = str(updated_record["_id"])
        
        log_database_operation(api_logger, "UPDATE", "emails", user_id, f"Updated status to {status} for email: {email}")
        log_email_operation(api_logger, "status_update", email, user_id, True, f"Status: {status}")
        
        return {
            "message": f"Email status updated to {status}",
            "email": email,
            "status": status,
            "updated_record": updated_record
        }
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error updating email status for {email}, user {user_id}: {e}",
                        extra={'user_id': user_id, 'email': email, 'error_type': 'status_update'})
        raise HTTPException(status_code=500, detail=f"Error updating email status: {str(e)}")

@app.patch("/emails/{user_id}/{email}")
def patch_email_fields(user_id: str, email: str, updates: dict):
    """Patch specific fields of an email record"""
    
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


## SMTP Credentials
@app.post("/smtp")
def set_smtp(smtp: SMTPCreds):
    try:
        api_logger.info(f"Setting SMTP credentials for user: {smtp.user_id}, host: {smtp.host}")
        
        smtp_col.replace_one({"user_id": smtp.user_id}, smtp.dict(), upsert=True)
        log_database_operation(api_logger, "UPSERT", "smtp_credentials", smtp.user_id, f"SMTP host: {smtp.host}")
        
        api_logger.info(f"Successfully configured SMTP for user: {smtp.user_id}")
        return smtp
        
    except Exception as e:
        api_logger.error(f"Failed to set SMTP credentials for user {smtp.user_id}: {e}",
                        extra={'user_id': smtp.user_id, 'error_type': 'smtp_config'})
        raise HTTPException(status_code=500, detail="Failed to configure SMTP")

@app.get("/smtp/{user_id}")
def get_smtp(user_id: str):
    try:
        api_logger.info(f"Fetching SMTP credentials for user: {user_id}")
        
        smtp = smtp_col.find_one({"user_id": user_id})
        if not smtp:
            api_logger.warning(f"SMTP config not found for user: {user_id}")
            raise HTTPException(status_code=404, detail="SMTP config not found")
            
        smtp["_id"] = str(smtp["_id"])  # Convert ObjectId to string
        log_database_operation(api_logger, "SELECT", "smtp_credentials", user_id, "SMTP config retrieved")
        
        return smtp
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to fetch SMTP config for user {user_id}: {e}",
                        extra={'user_id': user_id, 'error_type': 'smtp_retrieval'})
        raise HTTPException(status_code=500, detail="Failed to fetch SMTP config")



## Scheduler Collection
@app.post("/scheduler")
def set_scheduler(s: Scheduler):
    try:
        api_logger.info(f"Setting scheduler for user: {s.user_id}, interval: {s.interval}min, time: {s.time}")
        
        scheduler_col.replace_one({"user_id": s.user_id}, s.dict(), upsert=True)
        log_database_operation(api_logger, "UPSERT", "schedulers", s.user_id, 
                             f"Interval: {s.interval}min, Time: {s.time}, Max: {s.max_limit}")
        
        api_logger.info(f"Successfully configured scheduler for user: {s.user_id}")
        return s
        
    except Exception as e:
        api_logger.error(f"Failed to set scheduler for user {s.user_id}: {e}",
                        extra={'user_id': s.user_id, 'error_type': 'scheduler_config'})
        raise HTTPException(status_code=500, detail="Failed to configure scheduler")

@app.get("/scheduler/{user_id}")
def get_scheduler(user_id: str):
    try:
        api_logger.info(f"Fetching scheduler for user: {user_id}")
        
        sched = scheduler_col.find_one({"user_id": user_id})
        if not sched:
            api_logger.warning(f"Scheduler not found for user: {user_id}")
            raise HTTPException(status_code=404, detail="Scheduler not found")
            
        sched["_id"] = str(sched["_id"])
        log_database_operation(api_logger, "SELECT", "schedulers", user_id, "Scheduler config retrieved")
        
        return sched
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to fetch scheduler for user {user_id}: {e}",
                        extra={'user_id': user_id, 'error_type': 'scheduler_retrieval'})
        raise HTTPException(status_code=500, detail="Failed to fetch scheduler")




## Interviews Collection
@app.post("/interview/start")
def start_interview(user_id: str, email: EmailStr):
    try:
        api_logger.info(f"Starting interview for user: {user_id}, email: {email}")
        
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
        
        log_database_operation(api_logger, "INSERT", "interviews", user_id, f"Interview started for: {email}")
        api_logger.info(f"Successfully started interview for user: {user_id}, email: {email}, token: {token}")
        
        return interview
        
    except Exception as e:
        api_logger.error(f"Failed to start interview for user {user_id}, email {email}: {e}",
                        extra={'user_id': user_id, 'email': email, 'error_type': 'interview_start'})
        raise HTTPException(status_code=500, detail="Failed to start interview")

@app.post("/interview/complete/{token}")
def complete_interview(token: str):
    try:
        api_logger.info(f"Completing interview with token: {token}")
        
        result = interviews_col.find_one({"token": token})
        if not result:
            api_logger.warning(f"Interview not found for token: {token}")
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
        
        log_database_operation(api_logger, "UPDATE", "interviews", interview["user_id"], 
                             f"Interview completed for: {interview['email']}")
        api_logger.info(f"Successfully completed interview for token: {token}")
        
        return interview
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to complete interview for token {token}: {e}",
                        extra={'token': token, 'error_type': 'interview_completion'})
        raise HTTPException(status_code=500, detail="Failed to complete interview")

@app.get("/interview/status/{user_id}/{email}")
def check_interview_status(user_id: str, email: EmailStr):
    try:
        api_logger.info(f"Checking interview status for user: {user_id}, email: {email}")
        
        token = f"{user_id}-{email}"
        result = interviews_col.find_one({"token": token})
        
        status = result.get("status", "pending") if result else "not started"
        api_logger.info(f"Interview status for {email}: {status}")
        
        return {"status": status}
        
    except Exception as e:
        api_logger.error(f"Failed to check interview status for user {user_id}, email {email}: {e}",
                        extra={'user_id': user_id, 'email': email, 'error_type': 'interview_status_check'})
        raise HTTPException(status_code=500, detail="Failed to check interview status")

@app.get("/interview/{token}")
def get_interview(token: str):
    try:
        api_logger.info(f"Fetching interview details for token: {token}")
        
        interview = interviews_col.find_one({"token": token})
        if not interview:
            api_logger.warning(f"Interview not found for token: {token}")
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
        
        log_database_operation(api_logger, "SELECT", "interviews", interview["user_id"], f"Interview details retrieved")
        
        return interview
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to fetch interview for token {token}: {e}",
                        extra={'token': token, 'error_type': 'interview_retrieval'})
        raise HTTPException(status_code=500, detail="Failed to fetch interview")

@app.get("/interview/redirect/{token}")
def redirect_to_chat(token: str):
    """Redirect from old interview URL to new chat URL"""
    try:
        api_logger.info(f"Redirecting token {token} to chat interface")
        
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
        api_logger.error(f"Failed to redirect token {token}: {e}")
        raise HTTPException(status_code=500, detail="Failed to redirect to chat")

## Email Threads Collection
@app.get("/email-thread/{user_id}/{email}")
def get_email_thread(user_id: str, email: str):
    """Get email thread/conversation history for a specific email address"""
    try:
        api_logger.info(f"Fetching email thread for user: {user_id}, email: {email}")
        
        thread = list(email_contents_col.find(
            {"user_id": user_id, "email": email}
        ).sort("follow_up_number", 1))
        
        if not thread:
            api_logger.info(f"No email thread found for user: {user_id}, email: {email}")
            return {"email": email, "thread": [], "total_emails": 0}
        
        # Convert ObjectId to string
        for email_record in thread:
            email_record["_id"] = str(email_record["_id"])
        
        log_database_operation(api_logger, "SELECT", "email_contents", user_id, 
                             f"Retrieved thread with {len(thread)} emails for: {email}")
        
        return {
            "email": email,
            "thread": thread,
            "total_emails": len(thread),
            "last_sent": thread[-1]["created_at"] if thread else None
        }
        
    except Exception as e:
        api_logger.error(f"Failed to fetch email thread for user {user_id}, email {email}: {e}",
                        extra={'user_id': user_id, 'email': email, 'error_type': 'thread_retrieval'})
        raise HTTPException(status_code=500, detail="Failed to fetch email thread")

@app.get("/email-threads/{user_id}")
def get_all_email_threads(user_id: str):
    """Get all email threads for a user with summary"""
    try:
        api_logger.info(f"Fetching all email threads for user: {user_id}")
        
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
        
        log_database_operation(api_logger, "AGGREGATE", "email_contents", user_id, 
                             f"Retrieved {len(result)} thread summaries")
        api_logger.info(f"Successfully fetched {len(result)} email threads for user: {user_id}")
        
        return {"threads": result, "total_conversations": len(result)}
        
    except Exception as e:
        api_logger.error(f"Failed to fetch email threads for user {user_id}: {e}",
                        extra={'user_id': user_id, 'error_type': 'threads_retrieval'})
        raise HTTPException(status_code=500, detail="Failed to fetch email threads")


# Chat and Interview Processing Collection
## Interview Agent Chat (Streaming, with message count)
@app.post("/chat/interview/send/stream")
async def send_interview_message_stream(chat_msg: ChatMessage):
    """Send message to interview agent (streaming, tracks message count)"""
    try:
        api_logger.info(f"Sending interview message for user: {chat_msg.user_id}, session: {chat_msg.session_id}")
        
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
                "message_count": 1  # Start with 1 for the first message
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
                    "message_count": 1  # Increment message count for interviews
                }
            }
            
            chat_sessions_col.update_one(session_filter, session_update)
            
            # Get updated message count
            updated_session = chat_sessions_col.find_one(session_filter)
            message_count = updated_session.get("message_count", 1)
        
        log_database_operation(api_logger, "UPSERT", "chat_sessions", chat_msg.user_id, 
                             f"Interview message sent, session: {chat_msg.session_id}, count: {message_count}")
        
        # Use streaming response for interview agent (same as chat agent but with message count)
        async def generate_stream():
            try:
                # Send initial message count
                yield f"data: {json.dumps({'message_count': message_count, 'type': 'metadata'})}\n\n"
                
                # Prepare request to Lyzr streaming API
                headers = {
                    'Content-Type': 'application/json',
                    'x-api-key': api_key
                }
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
                                # Send completion signal with final message count
                                yield f"data: {json.dumps({'message_count': message_count, 'type': 'complete'})}\n\n"
                                yield f"data: [DONE]\n\n"
                                break
                            yield f"data: {data_content}\n\n"
                            await asyncio.sleep(0.01)  # Small delay for smooth streaming
                        
            except Exception as e:
                api_logger.error(f"Interview streaming error: {e}")
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
        api_logger.error(f"Failed to send interview message for user {chat_msg.user_id}: {e}",
                        extra={'user_id': chat_msg.user_id, 'error_type': 'interview_send'})
        raise HTTPException(status_code=500, detail="Failed to send interview message")

## Chat Agent (Streaming, no message count)
@app.post("/chat/agent/send/stream")
async def send_chat_agent_message_stream(chat_msg: ChatMessage):
    """Send message to chat agent with streaming response (no message count tracking)"""
    try:
        api_logger.info(f"Sending chat agent message for user: {chat_msg.user_id}, session: {chat_msg.session_id}")
        
        # Get user account to fetch API key and chat_agent_id
        user_account = accounts_col.find_one({"user_id": chat_msg.user_id})
        if not user_account:
            raise HTTPException(status_code=404, detail="User account not found")
        
        api_key = user_account.get("api_key")
        if not api_key:
            raise HTTPException(status_code=400, detail="No API key configured for this user")
        
        # Use chat_agent_id if available, otherwise fall back to agent_id
        actual_agent_id = user_account.get("chat_agent_id") or chat_msg.agent_id
        if not actual_agent_id:
            raise HTTPException(status_code=400, detail="No chat agent found. Please create a chat agent first.")
        
        # Create or update chat session record (NO message_count to avoid conflicts)
        session_filter = {
            "user_id": chat_msg.user_id,
            "session_id": chat_msg.session_id
        }
        
        session_update = {
            "$set": {
                "user_id": chat_msg.user_id,
                "session_id": chat_msg.session_id,
                "agent_id": actual_agent_id,
                "session_status": "active",
                "last_message_at": datetime.now(timezone.utc),
                "agent_type": "chat_agent"
            },
            "$setOnInsert": {
                "started_at": datetime.now(timezone.utc)
            }
        }
        
        chat_sessions_col.update_one(session_filter, session_update, upsert=True)
        
        log_database_operation(api_logger, "UPSERT", "chat_sessions", chat_msg.user_id, 
                             f"Chat agent session updated, session: {chat_msg.session_id}")
        
        # Use streaming response for chat agent
        async def generate_stream():
            try:
                # Prepare request to Lyzr streaming API
                headers = {
                    'Content-Type': 'application/json',
                    'x-api-key': api_key
                }
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
                api_logger.error(f"Chat agent streaming error: {e}")
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
        api_logger.error(f"Failed to send chat agent message for user {chat_msg.user_id}: {e}",
                        extra={'user_id': chat_msg.user_id, 'error_type': 'chat_agent_send'})
        raise HTTPException(status_code=500, detail="Failed to send chat agent message")

@app.get("/chat/history/{session_id}")
def get_chat_history_api(session_id: str, agent_id: str = None):
    """Get chat history for a session"""
    try:
        api_logger.info(f"Fetching chat history for session: {session_id}")
        
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
        
        return {
            "session_id": session_id,
            "messages": history,
            "total_messages": len(history) if history else 0
        }
        
    except Exception as e:
        api_logger.error(f"Failed to fetch chat history for session {session_id}: {e}",
                        extra={'session_id': session_id, 'error_type': 'chat_history'})
        raise HTTPException(status_code=500, detail="Failed to fetch chat history")



@app.post("/chat/interview/complete/{session_id}")
def complete_interview_by_session(session_id: str):
    """Complete interview by session_id - Public endpoint for interview participants"""
    try:
        api_logger.info(f"Completing interview by session: {session_id}")
        
        # Parse session_id to get user_id and email
        try:
            last_plus_index = session_id.rfind('+')
            if last_plus_index == -1:
                raise ValueError("Invalid session_id format")
            user_id = session_id[:last_plus_index]
            email = session_id[last_plus_index + 1:]
        except:
            api_logger.error(f"Invalid session_id format: {session_id}")
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
            api_logger.info(f"Created new interview record for session: {session_id}")
        else:
            # Update existing interview
            interviews_col.update_one(
                {"token": token},
                {"$set": {"status": "completed", "completed_at": datetime.utcnow()}}
            )
            api_logger.info(f"Updated existing interview record for session: {session_id}")
        
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
            api_logger.info(f"Created new chat session for session: {session_id}")
        else:
            # Update existing session
            chat_sessions_col.update_one(
                {"session_id": session_id},
                {"$set": {"session_status": "completed", "completed_at": datetime.utcnow()}}
            )
            api_logger.info(f"Updated existing chat session for session: {session_id}")
        
        log_database_operation(api_logger, "UPDATE", "interviews", user_id, 
                             f"Interview completed for session: {session_id}")
        
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
        api_logger.error(f"Failed to complete interview for session {session_id}: {e}",
                        extra={'session_id': session_id, 'error_type': 'interview_complete'})
        raise HTTPException(status_code=500, detail="Failed to complete interview")

@app.post("/interview/process")
def process_interview_completion(request: InterviewProcessRequest):
    """Process completed interview: generate PDF, upload to S3, parse and train KB"""
    try:
        api_logger.info(f"Processing interview completion for user: {request.user_id}, email: {request.email}")
        
        # Check if interview exists and is completed
        token = f"{request.user_id}-{request.email}"
        interview = interviews_col.find_one({"token": token})
        
        if not interview:
            api_logger.warning(f"Interview not found for processing: {token}")
            raise HTTPException(status_code=404, detail="Interview not found")
        
        # Generate session_id using the new format (user_id+email)
        session_id = f"{request.user_id}+{request.email}"
        
        # Get user account for API key
        user_account = accounts_col.find_one({"user_id": request.user_id})
        api_key = user_account.get("api_key") if user_account else None
        rag_id = request.rag_id or (user_account.get("rag_id") if user_account else None)
        
        # Process the completed interview
        result = process_completed_interview(
            user_id=request.user_id,
            email=request.email,
            rag_id=rag_id,
            api_key=api_key
        )
        
        # Check if there was an error in processing
        if result.get("error"):
            api_logger.warning(f"Interview processing had issues: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Update interview record with processing results BUT PRESERVE "completed" status
        current_status = interview.get("status", "completed")
        
        # Only change status to "processed" if it wasn't already "completed"
        # If user clicked "Complete Interview", status should remain "completed"
        if current_status != "completed":
            current_status = "processed"
        
        update_data = {
            "status": current_status,  # Preserve "completed" status if it exists
            "processed_at": datetime.utcnow(),
            "session_id": session_id,
            "pdf_generated": result.get("pdf_generated", False),
            "s3_upload_success": result.get("s3_upload_success", False)
        }
        
        # Add S3 URL if upload was successful
        if result.get("pdf_s3_url"):
            update_data["pdf_s3_url"] = result["pdf_s3_url"]
        
        # Add KB training info if available
        if request.rag_id:
            update_data["rag_id"] = request.rag_id
            update_data["kb_trained"] = result.get("kb_trained", False)
            if result.get("kb_error"):
                update_data["kb_error"] = result["kb_error"]
        
        interviews_col.update_one(
            {"token": token},
            {"$set": update_data}
        )
        
        # Update chat session if exists BUT PRESERVE "completed" status
        existing_chat_session = chat_sessions_col.find_one({"session_id": session_id})
        current_session_status = existing_chat_session.get("session_status", "completed") if existing_chat_session else "completed"
        
        # Only change status to "processed" if it wasn't already "completed"
        if current_session_status != "completed":
            current_session_status = "processed"
        
        chat_session_update = {
            "session_status": current_session_status,  # Preserve "completed" if it exists
            "processed_at": datetime.utcnow(),
            "pdf_generated": result.get("pdf_generated", False)
        }
        
        if result.get("pdf_s3_url"):
            chat_session_update["pdf_s3_url"] = result["pdf_s3_url"]
        
        chat_sessions_col.update_one(
            {"session_id": session_id},
            {"$set": chat_session_update}
        )
        
        # Create success message based on what was accomplished
        success_parts = []
        if result.get("pdf_generated"):
            success_parts.append("PDF generated")
        if result.get("s3_upload_success"):
            success_parts.append("uploaded to S3")
        if result.get("kb_trained"):
            success_parts.append("knowledge base trained")
        
        success_message = f"Interview processed successfully: {', '.join(success_parts)}"
        
        log_database_operation(api_logger, "UPDATE", "multiple", request.user_id, 
                             f"Interview processed: {request.email}, Results: {success_parts}")
        
        api_logger.info(f"Successfully processed interview for user: {request.user_id}, email: {request.email}")
        
        return {
            "message": success_message,
            "user_id": request.user_id,
            "email": request.email,
            "session_id": session_id,
            "interview_token": token,
            "processing_result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to process interview for user {request.user_id}, email {request.email}: {e}",
                        extra={'user_id': request.user_id, 'email': request.email, 'error_type': 'interview_processing'})
        raise HTTPException(status_code=500, detail=f"Failed to process interview: {str(e)}")


@app.get("/knowledge-base/pdfs/{user_id}")
def get_user_kb_pdfs(user_id: str):
    """Get all processed PDFs for a user's knowledge base"""
    try:
        api_logger.info(f"Fetching KB PDFs for user: {user_id}")

        chat_sessions = list(chat_sessions_col.find({
            "user_id": user_id,
            "pdf_s3_url": {"$exists": True, "$ne": None}
        }))

        interviews = list(interviews_col.find({
            "user_id": user_id,
            "pdf_s3_url": {"$exists": True, "$ne": None}
        }))
        
        pdfs = []
        seen_session_ids = set()
        
        for session in chat_sessions:
            email = session.get("email", "Unknown")
            session_id = session.get("session_id")
        
            if email == "Unknown" or not email or not session_id:
                continue

            if session_id in seen_session_ids:
                continue
                
            seen_session_ids.add(session_id)
            pdfs.append({
                "id": str(session.get("_id")),
                "email": email,
                "session_id": session_id,
                "pdf_url": session.get("pdf_s3_url"),
                "completed_at": session.get("processed_at", session.get("completed_at")),
                "message_count": session.get("message_count", 0),
                "type": "chat_session",
                "kb_trained": session.get("kb_trained", False)
            })

        for interview in interviews:
            email = interview.get("email", "Unknown")
            session_id = interview.get("session_id")
            
            if email == "Unknown" or not email or not session_id:
                continue

            if session_id in seen_session_ids:
                continue
                
            seen_session_ids.add(session_id)
            pdfs.append({
                "id": str(interview.get("_id")),
                "email": email,
                "session_id": session_id,
                "pdf_url": interview.get("pdf_s3_url"),
                "completed_at": interview.get("processed_at", interview.get("completed_at")),
                "message_count": interview.get("message_count", 0),
                "type": "interview",
                "kb_trained": interview.get("kb_trained", False)
            })
        
        # Sort by completion date (newest first)
        pdfs.sort(key=lambda x: x.get("completed_at") or "", reverse=True)
        
        api_logger.info(f"Found {len(pdfs)} PDFs for user {user_id}")
        
        return {
            "user_id": user_id,
            "pdf_count": len(pdfs),
            "pdfs": pdfs
        }
        
    except Exception as e:
        api_logger.error(f"Failed to fetch KB PDFs for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch PDFs: {str(e)}")

