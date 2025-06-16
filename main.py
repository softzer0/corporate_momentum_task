import datetime
import json
import logging
import os
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Generator, List, Literal, Optional

import requests
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Form, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get Hugging Face API configuration from environment
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "facebook/bart-large-cnn")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
USE_HUGGINGFACE_API = os.getenv("USE_HUGGINGFACE_API", "true").lower() == "true"

# Hugging Face API settings
HF_API_URL = f"https://api-inference.huggingface.co/models/{HUGGINGFACE_MODEL}"
HF_HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"} if HUGGINGFACE_API_TOKEN else None

app = FastAPI(
    title="Document Storage and Summarization API",
    description="A REST service for storing documents and generating summaries optimized for large documents",
    version="1.0.0",
)

# File-based storage configuration
DATA_DIR = Path("data")
DOCUMENTS_DIR = DATA_DIR / "documents"
SUMMARIES_DIR = DATA_DIR / "summaries"
METADATA_DIR = DATA_DIR / "metadata"

# Create directories
for directory in [DATA_DIR, DOCUMENTS_DIR, SUMMARIES_DIR, METADATA_DIR]:
    directory.mkdir(exist_ok=True)

# Thread lock for thread-safe operations
storage_lock = Lock()

# Constants for large file handling
MAX_CHUNK_SIZE = 8192  # 8KB chunks for reading large files
MAX_API_TEXT_LENGTH = 1000  # Max text length for Hugging Face API
STREAMING_THRESHOLD = 1024 * 1024  # 1MB - files larger than this will be streamed


# Pydantic models for API responses
class DocumentResponse(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the stored document")


class DocumentRetrievalResponse(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the document")
    text: str = Field(..., description="Full text content of the document")


class SummaryResponse(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the document")
    summary: str = Field(..., description="Generated summary of the document")


class SummaryStatusResponse(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the document")
    status: Literal["pending", "processing", "completed", "error", "stored"] = Field(
        ..., description="Current status of summary generation"
    )


class SummaryErrorResponse(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the document")
    error: str = Field(..., description="Error message for failed summary generation")


class DocumentMetadata(BaseModel):
    status: Literal["pending", "processing", "completed", "error", "stored"] = Field(
        default="pending", description="Document processing status"
    )
    created_at: Optional[str] = Field(default=None, description="Document creation timestamp")
    summary_status: Literal["pending", "processing", "completed", "error"] = Field(
        default="pending", description="Summary generation status"
    )
    file_size: Optional[int] = Field(default=None, description="File size in bytes")


class HuggingFaceApiResponseItem(BaseModel):
    summary_text: str = Field(..., description="Generated summary text from HuggingFace API")


class DocumentInfo(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the document")
    file_size: int = Field(..., description="Size of the document in bytes")
    status: Literal["pending", "processing", "completed", "error", "stored"] = Field(
        ..., description="Document storage status"
    )
    summary_status: Literal["pending", "processing", "completed", "error"] = Field(
        ..., description="Summary generation status"
    )
    created_at: Optional[str] = Field(None, description="Document creation timestamp (ISO format)")
    has_summary: bool = Field(..., description="Whether a summary has been generated")


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo] = Field(..., description="List of stored documents with metadata")
    total_count: int = Field(..., description="Total number of documents")


class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    documents_count: int = Field(..., description="Total number of stored documents")
    summaries_count: int = Field(..., description="Total number of generated summaries")
    api_available: bool = Field(..., description="Whether Hugging Face API token is available")


class RootResponse(BaseModel):
    message: str = Field(..., description="Welcome message")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error details")


# File-based storage utility functions
def get_document_path(document_id: str) -> Path:
    """Get the file path for a document"""
    return DOCUMENTS_DIR / f"{document_id}.txt"


def get_summary_path(document_id: str) -> Path:
    """Get the file path for a summary"""
    return SUMMARIES_DIR / f"{document_id}.txt"


def get_metadata_path(document_id: str) -> Path:
    """Get the file path for document metadata"""
    return METADATA_DIR / f"{document_id}.json"


def document_exists(document_id: str) -> bool:
    """Check if a document exists"""
    return get_document_path(document_id).exists()


def get_document_metadata(document_id: str) -> DocumentMetadata:
    """Get document metadata (status, timestamps, etc.)"""
    metadata_path = get_metadata_path(document_id)
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return DocumentMetadata(**data)
        except (json.JSONDecodeError, IOError):
            pass

    # Default metadata
    return DocumentMetadata()


def save_document_metadata(document_id: str, metadata: DocumentMetadata) -> None:
    """Save document metadata"""
    metadata_path = get_metadata_path(document_id)
    with storage_lock:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.model_dump(), f, indent=2)


def read_document_chunks(document_id: str) -> Generator[str, None, None]:
    """Generator to read document in chunks (for large files)"""
    document_path = get_document_path(document_id)
    if not document_path.exists():
        return

    with open(document_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(MAX_CHUNK_SIZE)
            if not chunk:
                break
            yield chunk


def read_document_text(document_id: str, max_length: Optional[int] = None) -> Optional[str]:
    """Read document text (with optional length limit for API calls)"""
    document_path = get_document_path(document_id)
    if not document_path.exists():
        return None

    if max_length:
        # Read only the specified amount for API calls
        with open(document_path, "r", encoding="utf-8") as f:
            return f.read(max_length)
    else:
        # Read full document
        with open(document_path, "r", encoding="utf-8") as f:
            return f.read()


def save_document_text(document_id: str, text: str) -> None:
    """Save document text to file"""
    document_path = get_document_path(document_id)
    with open(document_path, "w", encoding="utf-8") as f:
        f.write(text)


def save_summary(document_id: str, summary: str) -> None:
    """Save summary to file"""
    summary_path = get_summary_path(document_id)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)


def read_summary(document_id: str) -> Optional[str]:
    """Read summary from file"""
    summary_path = get_summary_path(document_id)
    if not summary_path.exists():
        return None

    with open(summary_path, "r", encoding="utf-8") as f:
        return f.read()


def get_all_document_ids() -> List[str]:
    """Get list of all document IDs"""
    return [f.stem for f in DOCUMENTS_DIR.glob("*.txt")]


def get_document_info(document_id: str) -> Optional[DocumentInfo]:
    """Get document information including size and metadata"""
    if not document_exists(document_id):
        return None

    document_path = get_document_path(document_id)
    metadata = get_document_metadata(document_id)

    # Get file size
    file_size = document_path.stat().st_size if document_path.exists() else 0

    return DocumentInfo(
        document_id=document_id,
        file_size=file_size,
        status=metadata.status,
        summary_status=metadata.summary_status,
        created_at=metadata.created_at,
        has_summary=get_summary_path(document_id).exists(),
    )


def query_huggingface_api(text: str) -> Optional[str]:
    """Query Hugging Face Inference API for summarization"""
    if not HUGGINGFACE_API_TOKEN or not HF_HEADERS:
        logger.warning("Hugging Face API token not configured")
        return None

    try:
        payload = {"inputs": text}
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=30)

        if response.status_code == 200:
            try:
                result = response.json()
                # Validate response structure and parse with Pydantic
                if isinstance(result, list) and result:
                    # Use Pydantic to validate and parse the response
                    api_response = HuggingFaceApiResponseItem.model_validate(result[0])
                    return api_response.summary_text
            except (ValueError, KeyError, IndexError, TypeError) as e:
                logger.error(f"Invalid JSON response from HuggingFace API: {e}")
        else:
            logger.error(f"HF API error: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error calling Hugging Face API: {e}")

    return None


def simple_extractive_summary(text: str, max_sentences: int = 3) -> str:
    """Simple extractive summarization as fallback"""
    import re

    # Split into sentences
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Take first few sentences as summary
    summary_sentences = sentences[:max_sentences]
    return ". ".join(summary_sentences) + "." if summary_sentences else text[:200] + "..."


def generate_summary_background(document_id: str) -> None:
    """Background task to generate summary using file-based storage"""
    try:
        # Update metadata to processing
        metadata = get_document_metadata(document_id)
        updated_metadata = DocumentMetadata(
            status=metadata.status,
            created_at=metadata.created_at,
            summary_status="processing",
            file_size=metadata.file_size,
        )
        save_document_metadata(document_id, updated_metadata)

        logger.info(f"Starting summarization for document {document_id}")

        # Read document text for summarization (limit for API)
        text_for_api = read_document_text(document_id, MAX_API_TEXT_LENGTH)
        if not text_for_api:
            raise ValueError("Document not found or empty")
        # Try Hugging Face API first (if API token is available)
        summary = None
        if HUGGINGFACE_API_TOKEN:
            summary = query_huggingface_api(text_for_api)

        # Fallback to simple extractive summary if API fails or no token
        if not summary:
            logger.info("Using fallback extractive summarization")
            # For fallback, read more text if available
            full_text = read_document_text(document_id)
            if full_text and len(full_text) > 500:
                summary = simple_extractive_summary(full_text, max_sentences=3)
            elif full_text:
                summary = simple_extractive_summary(full_text, max_sentences=2)
            else:
                summary = "Unable to generate summary: document not found"
        # Save summary and update metadata
        save_summary(document_id, summary)
        completed_metadata = DocumentMetadata(
            status=metadata.status,
            created_at=metadata.created_at,
            summary_status="completed",
            file_size=metadata.file_size,
        )
        save_document_metadata(document_id, completed_metadata)

        logger.info(f"Summary completed for document {document_id}")

    except Exception as e:
        logger.error(f"Error generating summary for {document_id}: {e}")
        # Update metadata to error status
        error_metadata = get_document_metadata(document_id)
        error_updated_metadata = DocumentMetadata(
            status=error_metadata.status,
            created_at=error_metadata.created_at,
            summary_status="error",
            file_size=error_metadata.file_size,
        )
        save_document_metadata(document_id, error_updated_metadata)
        save_summary(document_id, "Summary generation failed")


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """List all stored documents with their metadata.

    Returns:
        DocumentListResponse with list of documents and their information
    """
    try:
        document_ids = get_all_document_ids()
        documents_info: List[DocumentInfo] = []
        for doc_id in document_ids:
            info = get_document_info(doc_id)
            if info:
                documents_info.append(info)

        return DocumentListResponse(documents=documents_info, total_count=len(documents_info))

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/documents", response_model=DocumentResponse, status_code=201)
async def store_document(background_tasks: BackgroundTasks, request: Request) -> DocumentResponse:
    """
    Store a document via streaming upload (supports both small and large documents).
    Streams content directly to disk to handle very large documents without memory limitations.

    Args:
        request: Raw request with text content in the body

    Returns:
        DocumentResponse with document_id
    """
    try:
        # Generate unique document ID
        document_id = str(uuid.uuid4())[:8]  # Short UUID for simplicity
        document_path = get_document_path(document_id)

        # Stream the request content directly to disk
        total_size = 0

        logger.info(f"Starting streaming upload for document {document_id}")

        try:
            with open(document_path, "wb") as dest_file:
                # Read and write in chunks to handle large files efficiently
                async for chunk in request.stream():
                    # Decode chunk as UTF-8 text and re-encode to ensure proper text handling
                    try:
                        text_chunk = chunk.decode("utf-8")
                        dest_file.write(text_chunk.encode("utf-8"))
                        total_size += len(chunk)
                    except UnicodeDecodeError as decode_error:
                        # Clean up partial file on encoding error
                        dest_file.close()
                        document_path.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=400,
                            detail=f"Content contains invalid UTF-8 text: {decode_error}",
                        )

        except Exception as file_error:
            # Clean up partial file on any file operation error
            if document_path.exists():
                document_path.unlink(missing_ok=True)
            if isinstance(file_error, HTTPException):
                raise
            raise HTTPException(
                status_code=500, detail=f"Error processing uploaded content: {file_error}"
            )

        # Create metadata
        metadata = DocumentMetadata(
            status="stored",
            created_at=datetime.datetime.now().isoformat(),
            summary_status="pending",
            file_size=total_size,
        )
        save_document_metadata(document_id, metadata)  # Start background summarization
        background_tasks.add_task(generate_summary_background, document_id)

        logger.info(
            f"Streaming upload completed for document {document_id}, total size: {total_size} bytes"
        )
        return DocumentResponse(document_id=document_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/documents/form", response_model=DocumentResponse, status_code=201)
async def store_document_form(
    background_tasks: BackgroundTasks, text: str = Form(..., description="Text content to store")
) -> DocumentResponse:
    """
    Store a document via form-encoded data.
    Accepts a form field 'text' of arbitrary length.

    Args:
        text: Text content to store (from form field)

    Returns:
        DocumentResponse with document_id
    """
    try:
        # Generate unique document ID
        document_id = str(uuid.uuid4())[:8]  # Short UUID for simplicity

        # Save the text content
        save_document_text(document_id, text)

        # Create metadata
        metadata = DocumentMetadata(
            status="stored",
            created_at=datetime.datetime.now().isoformat(),
            summary_status="pending",
            file_size=len(text.encode("utf-8")),  # Size in bytes
        )
        save_document_metadata(document_id, metadata)

        # Start background summarization
        background_tasks.add_task(generate_summary_background, document_id)

        logger.info(f"Form document stored with ID {document_id}, size: {len(text)} characters")
        return DocumentResponse(document_id=document_id)

    except Exception as e:
        logger.error(f"Error storing form document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/documents/{document_id}")
async def get_document(document_id: str, accept: Optional[str] = Header(None)) -> Any:
    """
    Retrieve the full text of a document by document_id.
    Automatically streams for large documents or returns JSON for smaller ones.

    Response behavior:
    - Files > 1MB: Always streamed as text/plain
    - Files <= 1MB: JSON response unless Accept: text/plain header is provided
    - Client can force streaming by sending Accept: text/plain header

    Args:
        document_id: The unique identifier of the document
        accept: Accept header to specify preferred response format

    Returns:
        DocumentRetrievalResponse (JSON) for small files or StreamingResponse for large files
    """
    try:
        # Check if document exists
        if not document_exists(document_id):
            raise HTTPException(status_code=404, detail="Document not found")

        # Get file size to determine response type
        document_path = get_document_path(document_id)
        file_size = document_path.stat().st_size

        # Determine if we should stream
        should_stream = file_size > STREAMING_THRESHOLD or (
            accept and "text/plain" in accept.lower()
        )

        if should_stream:
            # Stream the file
            def generate_chunks() -> Generator[str, None, None]:
                """Generator to yield document chunks"""
                for chunk in read_document_chunks(document_id):
                    yield chunk

            return StreamingResponse(
                generate_chunks(),
                media_type="text/plain",
                headers={
                    "Content-Disposition": f"attachment; filename=document_{document_id}.txt",
                    "X-File-Size": str(file_size),
                    "X-Response-Type": "streamed",
                },
            )
        else:
            # Return JSON for smaller files
            text = read_document_text(document_id)
            if text is None:
                raise HTTPException(status_code=404, detail="Document not found")

            return DocumentRetrievalResponse(document_id=document_id, text=text)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/documents/{document_id}/summary", response_model=SummaryResponse)
async def get_document_summary(document_id: str) -> SummaryResponse:
    """
    Retrieve the summary of a document by document_id.

    Args:
        document_id: The unique identifier of the document

    Returns:
        SummaryResponse, SummaryStatusResponse, or SummaryErrorResponse depending on status
    """
    try:
        # Check if document exists
        if not document_exists(document_id):
            raise HTTPException(
                status_code=404, detail="Document not found"
            )  # Get document metadata
        metadata = get_document_metadata(document_id)
        summary_status = metadata.summary_status

        if summary_status == "pending":
            # If summary is still pending, start it if not already started
            background_tasks = BackgroundTasks()
            background_tasks.add_task(generate_summary_background, document_id)
            raise HTTPException(
                status_code=202,
                detail="Summary generation in progress. Please try again in a moment.",
            )
        elif summary_status == "processing":
            raise HTTPException(
                status_code=202,
                detail="Summary generation in progress. Please try again in a moment.",
            )
        elif summary_status == "error":
            raise HTTPException(status_code=500, detail="Summary generation failed")
        elif summary_status == "completed":
            summary = read_summary(document_id)
            if summary:
                return SummaryResponse(document_id=document_id, summary=summary)

        # Fallback: generate summary synchronously
        text = read_document_text(document_id)
        if text:
            summary = simple_extractive_summary(text)
            save_summary(document_id, summary)

            # Update metadata
            updated_metadata = DocumentMetadata(
                status=metadata.status,
                created_at=metadata.created_at,
                summary_status="completed",
                file_size=metadata.file_size,
            )
            save_document_metadata(document_id, updated_metadata)

            return SummaryResponse(document_id=document_id, summary=summary)
        else:
            raise HTTPException(status_code=404, detail="Document not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving summary for {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/", response_model=RootResponse)
async def root() -> RootResponse:
    """Health check endpoint"""
    return RootResponse(message="Document Storage and Summarization API is running")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Detailed health check"""
    document_ids = get_all_document_ids()
    summaries_count = sum(1 for doc_id in document_ids if get_summary_path(doc_id).exists())

    return HealthResponse(
        status="healthy",
        documents_count=len(document_ids),
        summaries_count=summaries_count,
        api_available=bool(HUGGINGFACE_API_TOKEN),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # pyright: ignore[reportUnknownMemberType]
