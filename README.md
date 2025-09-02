# Marketing Agent Backend API

This is the backend API for the Marketing Agent application that handles email campaigns, chat interviews, and knowledge base integration.

## New Features

### Chat-Based Interviews
The system now supports chat-based interviews using Lyzr's conversational AI:

- **Session ID Format**: `{user_id}+{email}` (e.g., "user123+john@example.com")
- **Chat Interface**: Interviews now use a conversational chat interface instead of static forms
- **Real-time Processing**: Chat messages are sent to Lyzr's inference API in real-time
- **Automatic PDF Generation**: Chat conversations are automatically converted to PDFs
- **Knowledge Base Integration**: PDFs are parsed and added to RAG knowledge bases

### Key Endpoints

#### Chat Management
- `POST /chat/send` - Send a message to the chat interface
- `GET /chat/history/{session_id}` - Get chat history for a session
- `GET /chat/sessions/{user_id}` - Get all chat sessions for a user
- `POST /chat/session/complete/{session_id}` - Mark chat session as completed

#### Interview Processing
- `POST /interview/process` - Process completed interview (generate PDF, upload to S3, train KB)
- `GET /interview/chat-link/{user_id}/{email}` - Get chat interface link for an interview

### Workflow

1. **Email Campaigns**: Users receive emails with links to chat interfaces instead of static interview pages
2. **Chat Interface**: Recipients click the link and engage in a conversation with the AI agent
3. **Real-time Chat**: Messages are sent to Lyzr's chat API and responses are returned
4. **Session Tracking**: Chat sessions are tracked with status (active, completed, processed)
5. **Completion Processing**: When a chat is completed, the system:
   - Retrieves the full chat history from Lyzr API
   - Generates a PDF document from the conversation
   - Uploads the PDF to AWS S3
   - Parses the PDF using Lyzr's parse API
   - Trains the RAG knowledge base with the parsed content

### Configuration

Create a `.env` file based on `.env.example` with the following configurations:

```env
# MongoDB
MONGODB_URL=mongodb://localhost:27017

# AWS S3 (for PDF storage)
AWS_S3_BUCKET=your-bucket-name
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1

# Lyzr API
LYZR_API_KEY=your-lyzr-api-key
```

### Dependencies

New dependencies added for this functionality:
- `reportlab` - PDF generation
- `boto3` - AWS S3 integration
- `aiofiles` - Async file operations

### API Integration

The system integrates with several Lyzr APIs:

1. **Chat API**: `https://agent-prod.studio.lyzr.ai/v3/inference/chat/`
2. **History API**: `https://agent-prod.studio.lyzr.ai/v1/sessions/{session_id}/history`
3. **Parse PDF API**: `https://rag-prod.studio.lyzr.ai/v3/parse/pdf/`
4. **Train RAG API**: `https://rag-prod.studio.lyzr.ai/v3/rag/train/{rag_id}/`

### Database Collections

New MongoDB collections:
- `chat_sessions` - Tracks chat sessions and their status
- Extended `interviews` collection with processing status and PDF URLs

### PDF Generation

Chat conversations are converted to professional PDF documents containing:
- Chat participants (User and Assistant)
- Message timestamps
- Full conversation history
- Metadata (user ID, email, session details)

### Knowledge Base Training

After PDF generation:
1. PDF is uploaded to AWS S3 for permanent storage
2. PDF is sent to Lyzr's parse API for content extraction
3. Parsed content is sent to the RAG training API
4. The conversation becomes part of the searchable knowledge base

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env`

3. Start the server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

4. Start the scheduler (for automated emails):
```bash
python main.py
```

### Frontend Integration

The frontend should now direct users to chat interfaces at:
`http://localhost:3000/chat/{session_id}`

Where `session_id` follows the format: `{user_id}+{email}`

### Example Usage

1. **Start an Interview**:
```bash
POST /interview/start
{
  "user_id": "user123",
  "email": "john@example.com"
}
```

2. **Get Chat Link**:
```bash
GET /interview/chat-link/user123/john@example.com?agent_id=agent456
```

3. **Send Chat Message**:
```bash
POST /chat/send
{
  "user_id": "user123",
  "agent_id": "agent456",
  "session_id": "user123+john@example.com",
  "message": "Hello, I'm here for the interview"
}
```

4. **Process Completed Interview**:
```bash
POST /interview/process
{
  "user_id": "user123",
  "email": "john@example.com",
  "rag_id": "rag789"
}
```

This will generate a PDF, upload it to S3, and train the knowledge base automatically.

This project is a FastAPI application designed to manage email scheduling for interviews. It includes endpoints for managing accounts, emails, SMTP configurations, schedulers, and interviews. The application connects to a MongoDB database and utilizes Pydantic for data validation.

## Project Structure

```
fastapi-email-scheduler
├── app.py                # FastAPI application with various endpoints
├── main.py               # Scheduler logic for sending follow-up emails
├── requirements.txt      # List of dependencies
├── .env                  # Environment variables
├── .gitignore            # Files and directories to ignore by Git
├── README.md             # Project documentation
└── tests
    └── test_basic.py     # Unit tests for the application
```

## Setup Instructions

1. **Install Dependencies**  
   Install the required dependencies by running:

   ```
   pip install -r requirements.txt
   ```

2. **Run the FastAPI Application**  
   Start the FastAPI application by running:

   ```
   uvicorn app:app --reload
   ```

   This command tells Uvicorn to run the FastAPI app defined in `app.py`.

3. **Access Swagger UI**  
   Once the server is running, you can access the Swagger UI by navigating to:

   ```
   http://127.0.0.1:8000/docs
   ```

   This URL will display the interactive API documentation generated by FastAPI.

4. **Run the Scheduler**  
   If you want to run the scheduler defined in `main.py`, you can execute it in a separate terminal:
   ```
   python main.py
   ```
   This will start the email scheduling process as defined in the `main.py` file.

## Usage

- Use the endpoints defined in `app.py` to manage accounts, emails, SMTP configurations, and interviews.
- The scheduler in `main.py` will automatically send follow-up emails based on user configurations stored in the MongoDB database.

## Testing

Unit tests can be found in the `tests/test_basic.py` file. You can run these tests to ensure the functionality of the application.
