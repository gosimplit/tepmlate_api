from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import create_client, Client
from pydantic import BaseModel
import os
import json
import re
import logging
import asyncio
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import io
from docx import Document
import PyPDF2
from openai import OpenAI

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


security = HTTPBearer()

# Initialize Supabase client with service role key (server-side only!)
supabase_url = os.getenv("SUPABASE_URL")
supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not supabase_url or not supabase_service_role_key:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in environment variables")
supabase: Client = create_client(supabase_url, supabase_service_role_key)

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY must be set in environment variables")
openai_client = OpenAI(api_key=openai_api_key)


class ProcessTemplateRequest(BaseModel):
    template_id: str


class ProcessTemplateResponse(BaseModel):
    success: bool
    template_id: str


async def validate_user(jwt_token: str):
    """Validate JWT and get user info using Supabase auth"""
    try:
        # This validates the JWT and returns user info
        user_response = supabase.auth.get_user(jwt_token)
        
        if user_response.user is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return user_response.user
    except Exception as e:
        # Handle Supabase auth errors
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=401, detail=f"Token validation failed: {str(e)}")


def parse_document(file_bytes: bytes, mime_type: str) -> str:
    """Parse document based on mime type and return text content"""
    try:
        if mime_type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        elif mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                          "application/msword"]:
            doc = Document(io.BytesIO(file_bytes))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        
        elif mime_type == "text/plain":
            return file_bytes.decode("utf-8")
        
        else:
            # Try to decode as UTF-8 text
            try:
                return file_bytes.decode("utf-8")
            except UnicodeDecodeError as decode_error:
                logger.error(f"Failed to decode file as UTF-8: {decode_error}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {mime_type}"
                )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error parsing document: {str(e)}"
        )


async def generate_template_sections(
    document_text: str,  
    preferences_context: str
) -> Dict[str, Any]:
    """Generate template sections using AI based on document and preferences"""
    try:
        # Create prompt for AI
        prompt = f"""Analyze the following document and generate structured sections based following the user's preferences.

{preferences_context}

Document Content:
{document_text}

Please generate a structured JSON object with sections that organizes this document content. Each section has a name, description, and a list of requisites. Each section can represent one or multiple paragraphs of a document. Each section must have a purpose. If multiple paragraphs are for diferent purposes, divide them into sections. do not make multiple sections for the same paragraph.
  Return only valid JSON with a structure:
{{
    
    "template" : {{
        "structure" : {{
            "sections" : [
                {{
                    "sectionName": {{
                        "description": "string",
                        "requisites": [
                            "string",
                            "string"
                        ]
                    }}
                }},
                {{
                    "sectionName": {{
                        "description": "string",
                        "requisites": [
                            "string",
                            "string"
                        ]
                    }}
                }}
            ]
        }}
    }}
}}

"""

        # Call OpenAI API with timeout
        logger.info("Calling OpenAI API to generate template sections")
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    openai_client.chat.completions.create,
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a document analysis assistant that generates structured sections from documents. Follow the provided JSON structure and generate the sections. Follow the following steps: 1. provide you initial answer 2.Generate verification 3-5 checks to expose errors in you answer, 4. fix each error independently, 5. provide the final revised answer based on the fixes."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=10000
                ),
                timeout=300.0  # 5 minute timeout
            )
            logger.info("OpenAI API call completed successfully")
        except asyncio.TimeoutError:
            logger.error("OpenAI API call timed out after 5 minutes")
            raise HTTPException(status_code=504, detail="AI processing timed out")
        except Exception as ai_error:
            logger.error(f"OpenAI API error: {ai_error}", exc_info=True)
            raise HTTPException(status_code=500, detail="AI processing failed")
        
        # Extract and parse JSON from response
        ai_response = response.choices[0].message.content
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
        if json_match:
            sections_data = json.loads(json_match.group())
            return sections_data
        else:
            # Fallback: create a simple structure
            return {
                "sections": [
                    {
                        "title": "Document Content",
                        "content": document_text[:500] + "..." if len(document_text) > 500 else document_text,
                        "order": 1
                    }
                ]
            }
    
    except Exception as e:
        # Fallback structure if AI fails
        return {
            "sections": [
                {
                    "title": "Document Content",
                    "content": document_text[:500] + "..." if len(document_text) > 500 else document_text,
                    "order": 1
                }
            ],
            "error": str(e)
        }


@app.post("/process-template", response_model=ProcessTemplateResponse)
async def process_template(
    request: ProcessTemplateRequest,
    authorization: str = Header(None)
):
    """
    Process a template: extract user context, get document content,
    generate sections using AI, and update the template in database.
    """
    logger.info(f"Processing template request for template_id: {request.template_id}")
    try:
        # 1. Extract and validate user from JWT
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header missing")
        
        # Remove "Bearer " prefix
        jwt_token = authorization.replace("Bearer ", "")
        
        # Validate JWT and get user
        user = await validate_user(jwt_token)
        user_id = user.id
        logger.info(f"User validated: {user_id}")
        
        # 2. Validate template_id
        if not request.template_id or len(request.template_id.strip()) == 0:
            raise HTTPException(status_code=400, detail="template_id is required")
        if len(request.template_id) < 10:  # Basic format check
            raise HTTPException(status_code=400, detail="Invalid template_id format")
        
        # 3. Fetch template record
        template_response = supabase.table("templates_data").select("*").eq("id", request.template_id).execute()
        
        if not template_response.data:
            logger.warning(f"Template not found: {request.template_id}")
            raise HTTPException(status_code=404, detail="Template not found")
        
        template = template_response.data[0]
        
        # 4. Fetch user context
       
        preferences_response = supabase.table("user_preferences").select("language").eq("user_id", user_id).execute()
        
        preferences = preferences_response.data[0] if preferences_response.data else {}
        
        # Build context strings from profile and preferences (before expensive document processing)
        preferences_context = f"Preferences: {preferences}"
        
        # 5. Get document content
        if template.get("template_content"):
            # Text was pasted directly
            document_text = template["template_content"]
        else:
            # File was uploaded - fetch from storage
            file_response = supabase.table("template_files").select("*").eq("template_id", request.template_id).execute()
            
            if not file_response.data:
                raise HTTPException(status_code=404, detail="Template file not found")
            
            file_record = file_response.data[0]
            file_path = file_record["file_path"]
            mime_type = file_record.get("mime_type", "text/plain")
            
            # Download file from storage
            try:
                logger.info(f"Downloading file from storage: {file_path}")
                # Supabase storage download returns bytes directly
                file_bytes = supabase.storage.from_("template-files").download(file_path)
                if not isinstance(file_bytes, bytes):
                    # Convert to bytes if needed
                    file_bytes = bytes(file_bytes) if file_bytes else b""
                logger.info(f"File downloaded successfully, size: {len(file_bytes)} bytes")
            except Exception as storage_error:
                logger.error(f"Error downloading file from storage: {storage_error}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Error downloading file from storage: {str(storage_error)}"
                )
            document_text = parse_document(file_bytes, mime_type)
        
        # 6. Process and generate sections (AI magic here)
        sections = await generate_template_sections(document_text, preferences_context)
        
        # 7. Update template in database
        logger.info(f"Updating template in database: {request.template_id}")
        update_data = {
            "template_data": sections
        }
        
        # Only update template_content if it was from file
        if not template.get("template_content"):
            update_data["template_content"] = document_text
        
        supabase.table("templates_data").update(update_data).eq("id", request.template_id).execute()
        logger.info(f"Template updated successfully: {request.template_id}")
        
        # 8. Return success
        return ProcessTemplateResponse(
            success=True,
            template_id=request.template_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal server error: {str(e)}", exc_info=True)
        # Only expose details in development
        if os.getenv("ENVIRONMENT") == "development":
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail="An internal error occurred")


@app.get("/")
async def root():
    return {"message": "Template Processing API is running", "status": "healthy"}


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Quick check if Supabase is accessible
        supabase.table("templates_data").select("id").limit(1).execute()
        return {
            "status": "healthy",
            "services": {
                "supabase": "connected",
                "openai": "configured"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Service unhealthy")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

