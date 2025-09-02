import os
import time
import smtplib
import requests
import os
import time
import smtplib
import requests
from email.message import EmailMessage
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from logger_config import (
    scheduler_logger, 
    email_logger,
    log_email_operation,
    log_scheduler_event
)
load_dotenv()

URL = os.getenv("MONGODB_URL")

try:
    client = MongoClient(URL)
    db = client.data_collection_agent
    emails_col = db.emails
    scheduler_col = db.schedulers
    smtp_col = db.smtp_credentials
    interviews_col = db.interviews
    email_contents_col = db.email_contents
    
    # Test connection
    client.admin.command('ping')
    scheduler_logger.info("Successfully connected to MongoDB for scheduler", extra={'database': 'data_collection_agent'})
    
except Exception as e:
    scheduler_logger.critical(f"Failed to connect to MongoDB: {e}", extra={'error_type': 'database_connection'})
    raise

API_BASE_URL = os.getenv("API_BASE_URL") or "http://localhost:8000"

def update_email_status_via_api(user_id, email_address, status, error_message=None):
    """Update email status via FastAPI endpoint"""
    try:
        url = f"{API_BASE_URL}/emails/{user_id}/{email_address}/status"
        params = {"status": status}
        if error_message:
            params["error_message"] = error_message
            
        response = requests.put(url, params=params)
        return response.status_code == 200
    except Exception as e:
        scheduler_logger.error(f"API update failed for {email_address}: {e}")
        return False

# Send email via SMTP
def send_email_smtp(to_email, subject, body, smpt_config, user_id=None):
    msg = EmailMessage()
    
    # Validate SMTP configuration
    if not smpt_config:
        if user_id:
            update_email_status_via_api(user_id, to_email, "failed", "No SMTP configuration found")
        return False
        
    SMTP_USERNAME = smpt_config.get("username")
    SMTP_PASSWORD = smpt_config.get("password")
    SMTP_HOST = smpt_config.get("host")
    SMTP_PORT = 587
    
    if not all([SMTP_USERNAME, SMTP_PASSWORD, SMTP_HOST]):
        if user_id:
            update_email_status_via_api(user_id, to_email, "failed", "Incomplete SMTP configuration")
        return False
    
    msg["Subject"] = subject
    msg["From"] = SMTP_USERNAME
    msg["To"] = to_email
    msg.set_content(body)
    
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        
        log_email_operation(email_logger, "send", to_email, user_id, True, f"Subject: {subject}")
        
        if user_id:
            update_email_status_via_api(user_id, to_email, "sent")
        
        return True
        
    except smtplib.SMTPRecipientsRefused as e:
        error_msg = f"Recipient refused: {e}"
        log_email_operation(email_logger, "send", to_email, user_id, False, error_msg)
        if user_id:
            update_email_status_via_api(user_id, to_email, "bounced", error_msg)
        return False
        
    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"SMTP Authentication failed: {e}"
        log_email_operation(email_logger, "send", to_email, user_id, False, error_msg)
        if user_id:
            update_email_status_via_api(user_id, to_email, "failed", error_msg)
        return False
        
    except Exception as e:
        error_msg = f"Email sending failed: {e}"
        log_email_operation(email_logger, "send", to_email, user_id, False, error_msg)
        if user_id:
            update_email_status_via_api(user_id, to_email, "failed", error_msg)
        return False

def make_utc(dt):
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt

def get_email_content(user_id, email_address, follow_up_num, interview_link, interview_status):
    """Get email content using default templates"""
    
    # Use default email templates based on follow-up number and status
    if interview_status == "started":
        # User has started but not completed the interview
        subject = "Please Complete Your Interview"
        body = f"""Hi,

Thanks for starting your interview. Just a reminder to complete it here:
{interview_link}

Let us know if you have any questions.

Best regards,
The Hiring Team"""
    else:
        # User hasn't started the interview yet (pending) - use normal follow-up sequence
        if follow_up_num == 1:
            subject = "You're invited for an interview!"
            body = f"""Hi,

We're excited to move forward with your application. Please complete your interview here:
{interview_link}

This interview should take about 15-20 minutes to complete.

Thanks!
The Hiring Team"""
        elif follow_up_num == 2:
            subject = "Friendly Reminder: Your Interview Awaits"
            body = f"""Hi again,

Just checking in — we'd still love to hear from you. Please complete your interview here:
{interview_link}

If you have any questions or need assistance, please don't hesitate to reach out.

Best regards,
The Hiring Team"""
        elif follow_up_num == 3:
            subject = "Final Reminder: Interview Opportunity Closing Soon"
            body = f"""Hi,

This is our final reminder about your interview opportunity. If you're still interested, please complete your interview here:
{interview_link}

We'll be closing this opportunity soon, so please respond at your earliest convenience.

Thanks again for your time!
The Hiring Team"""
        else:
            subject = f"Follow-up #{follow_up_num} - Interview Invitation"
            body = f"""Hi,

This is follow-up #{follow_up_num} regarding your interview opportunity.

Please complete your interview here:
{interview_link}

We look forward to hearing from you.

Best regards,
The Hiring Team"""
    
    return subject, body

def record_sent_email(user_id, email_address, subject, body, follow_up_num, status="sent"):
    """Record the email that was sent in the email_contents collection"""
    email_record = {
        "user_id": user_id,
        "email": email_address,
        "subject": subject,
        "content": body,
        "follow_up_number": follow_up_num,
        "email_status": status,
        "created_at": datetime.now(timezone.utc)
    }
    
    try:
        email_contents_col.insert_one(email_record)
    except Exception as e:
        scheduler_logger.error(f"Failed to record email: {e}")

# New function to auto-start interviews at scheduled time
def auto_start_interviews_if_needed():
    """Check if it's time to start interviews for any user and auto-start them"""
    now = datetime.now(timezone.utc)
    local_now = datetime.now()
    
    # Get all schedulers
    schedulers = list(scheduler_col.find({}))
    
    if not schedulers:
        return
    
    for scheduler in schedulers:
        user_id = scheduler["user_id"]
        scheduled_time = scheduler.get("time", "14:10")  # e.g., "14:30"
        
        # Parse scheduled time (format: "HH:MM")
        try:
            scheduled_hour, scheduled_minute = map(int, scheduled_time.split(":"))
        except:
            scheduler_logger.error(f"Invalid time format for user {user_id}: {scheduled_time}")
            continue
            
        # Use LOCAL time for comparison
        current_hour = local_now.hour
        current_minute = local_now.minute
        
        # Check if we're within 5 minutes of scheduled time
        current_time_minutes = current_hour * 60 + current_minute
        scheduled_time_minutes = scheduled_hour * 60 + scheduled_minute
        time_diff_minutes = abs(current_time_minutes - scheduled_time_minutes)
        
        if time_diff_minutes > 5:  # More than 5 minutes difference
            continue
            
        # Check if daily trigger already happened today
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Check if interviews were already triggered today by looking for a specific marker
        daily_trigger_marker = scheduler_col.find_one({
            "user_id": user_id,
            "last_daily_trigger": {"$gte": today_start}
        })
        
        if daily_trigger_marker:
            continue  # Already triggered today
            
        scheduler_logger.info(f"Daily email trigger for user {user_id} at {scheduled_time}")
        
        # Get all pending emails for this user that were uploaded BEFORE today's trigger time
        # Only process emails that were uploaded before the scheduled time today
        today_trigger_time = now.replace(
            hour=scheduled_hour, 
            minute=scheduled_minute, 
            second=0, 
            microsecond=0
        )
        
        pending_emails = list(emails_col.find({
            "user_id": user_id,
            "status": {"$ne": "exhausted"},
            "created_at": {"$lt": today_trigger_time}  # Only emails uploaded before today's trigger
        }))
        
        if not pending_emails:
            # Mark that daily trigger happened even if no emails to process
            scheduler_col.update_one(
                {"user_id": user_id},
                {"$set": {"last_daily_trigger": now}},
                upsert=True
            )
            continue
            
        # Auto-start interviews for eligible emails
        interviews_started = 0
        for email_record in pending_emails:
            email_address = email_record["email"]
            
            # Check if interview already exists for this email
            existing_interview = interviews_col.find_one({
                "user_id": user_id,
                "email": email_address
            })
            
            if existing_interview:
                continue  # Skip if interview already exists
                
            # Create interview record
            token = f"{user_id}-{email_address}"
            interview = {
                "user_id": user_id,
                "email": email_address,
                "token": token,
                "status": "pending",
                "created_at": now
            }
            
            interviews_col.insert_one(interview)
            interviews_started += 1
            
        # Mark that daily trigger happened for this user
        scheduler_col.update_one(
            {"user_id": user_id},
            {"$set": {"last_daily_trigger": now}},
            upsert=True
        )
        
        log_scheduler_event(scheduler_logger, "daily_trigger", 
                           details=f"User {user_id}: {interviews_started} interviews started")

# Scheduler logic
def run_scheduler():
    now = datetime.now(timezone.utc)
    
    # First, check if we need to auto-start any interviews
    auto_start_interviews_if_needed()
    
    # Then process existing interviews for email sending
    interviews = db.interviews.find({
        "status": {"$nin": ["completed", "processed"]}  # Skip both completed and processed interviews
    })

    emails_sent = 0
    
    for interview in interviews:
        user_id = interview["user_id"]
        email_address = interview["email"]
        token = interview["token"]
        interview_status = interview.get("status", "pending")

        # Skip if interview is completed or processed
        if interview_status in ["completed", "processed"]:
            continue

        # Also check if chat session exists and is completed
        session_id = f"{user_id}+{email_address}"
        chat_session = db.chat_sessions.find_one({"session_id": session_id})
        if chat_session and chat_session.get("session_status") in ["completed", "processed"]:
            # Update interview status to match chat session if not already completed/processed
            if interview_status not in ["completed", "processed"]:
                new_status = chat_session.get("session_status")
                interviews_col.update_one(
                    {"token": token},
                    {"$set": {"status": new_status, "completed_at": datetime.now(timezone.utc)}}
                )
            continue

        # Get scheduler for user
        scheduler = db.schedulers.find_one({"user_id": user_id})
        if not scheduler:
            continue

        interval_minutes = scheduler["interval"]
        max_limit = scheduler["max_limit"]

        # Get email record
        email = emails_col.find_one({"user_id": user_id, "email": email_address})
        if not email:
            continue

        follow_up_count = email.get("follow_up_count", 0)
        last_sent_at = email.get("last_sent_at")

        # Check if email is eligible for follow-up
        if follow_up_count >= max_limit:
            continue

        if last_sent_at and make_utc(last_sent_at) > now - timedelta(minutes=interval_minutes):
            continue
        
        # Prepare email content
        follow_up_num = follow_up_count + 1
        
        # Get user account to fetch agent_id
        from pymongo import MongoClient
        accounts_col = db.accounts
        user_account = accounts_col.find_one({"user_id": user_id})
        agent_id = user_account.get("agent_id") if user_account else None
        
        # Generate session_id for the new chat interface
        session_id = f"{user_id}+{email_address}"
        
        # URL encode session_id for use as path parameter
        import urllib.parse
        encoded_session_id = urllib.parse.quote(session_id, safe='')
        
        # Generate interview link with encoded session_id as path parameter and agent_id as query parameter
        if agent_id:
            interview_link = f"http://localhost:3000/chat/{encoded_session_id}?agent_id={agent_id}"
        else:
            # Fallback if no agent_id
            interview_link = f"http://localhost:3000/chat/{encoded_session_id}"
        
        subject, body = get_email_content(
            user_id=user_id,
            email_address=email_address,
            follow_up_num=follow_up_num,
            interview_link=interview_link,
            interview_status=interview_status
        )

        # Get SMTP credentials for the user
        smtp_config = smtp_col.find_one({"user_id": user_id})
        if not smtp_config:
            continue

        # Send email using user's SMTP credentials
        success = send_email_smtp(
            to_email=email_address,
            subject=subject,
            body=body,
            smpt_config=smtp_config,
            user_id=user_id
        )

        if success:
            emails_sent += 1
            
            # Record the sent email in email_contents collection
            record_sent_email(
                user_id=user_id,
                email_address=email_address,
                subject=subject,
                body=body,
                follow_up_num=follow_up_num,
                status="sent"
            )
            
            # Update email record with follow-up count and last sent time
            update_fields = {
                "last_sent_at": now,
                "follow_up_count": follow_up_num,
                "updated_at": now
            }

            if follow_up_num >= max_limit:
                update_fields["status"] = "exhausted"
            else:
                update_fields["status"] = "sent"

            emails_col.update_one(
                {"_id": email["_id"]},
                {"$set": update_fields}
            )
        else:
            # Record the failed email attempt
            record_sent_email(
                user_id=user_id,
                email_address=email_address,
                subject=subject,
                body=body,
                follow_up_num=follow_up_num,
                status="failed"
            )

    if emails_sent > 0:
        scheduler_logger.info(f"Emails sent: {emails_sent}")

# Entry point
if __name__ == "__main__":
    scheduler_logger.info("Starting Marketing Agent Email Scheduler")
    
    while True:
        try:
            run_scheduler()
        except Exception as e:
            scheduler_logger.error(f"Scheduler error: {e}")
        
        time.sleep(300)  # run every 5 minutes
