from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter, Form
from typing import Optional
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from backend.schemas.model import DocumentUploadResponse
from backend.services.text_processing import  ChunkingService
from backend.services.services import db_manager, vector_store, embedding_service, text_extractor, settings
import logging
import uuid


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ingestion"])

@router.post(
    "/api/v1/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=201
)
async def upload_document(
    file: UploadFile = File(..., description="PDF or TXT file to upload"),
    chunking_strategy: str = Form(
        default="fixed",
        description="Chunking strategy: 'fixed' or 'semantic'"
    ),
    chunk_size: Optional[int] = Form(
        default=None,
        description="Custom chunk size (optional)"
    ),
    chunk_overlap: Optional[int] = Form(
        default=None,
        description="Custom chunk overlap (optional)"
    )
) -> DocumentUploadResponse:
    """
    Upload and process a document.
    
    - Extracts text from PDF or TXT
    - Chunks text using selected strategy
    - Generates embeddings
    - Stores in vector database and metadata DB
    """
    
    # Validate file type
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(
            status_code=400,
            detail="Only .pdf and .txt files are supported"
        )
    
    # Validate file size
    file_content = await file.read()
    file_size_mb = len(file_content) / (1024 * 1024)
    
    if file_size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum allowed size of {settings.max_file_size_mb}MB"
        )
    
    try:
        # Extract text
        logger.info(f"Extracting text from {file.filename}")
        if file.filename.endswith('.pdf'):
            text = await text_extractor.extract_from_pdf(file_content)
            file_type = "pdf"
        else:
            text = await text_extractor.extract_from_txt(file_content)
            file_type = "txt"
        
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the file"
            )
        
        # Chunk text
        logger.info(f"Chunking text using {chunking_strategy} strategy")
        chunking_service = ChunkingService(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap
        )
        chunks = await chunking_service.chunk_text(text, chunking_strategy)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Text chunking produced no chunks"
            )
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        print(document_id)
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = await embedding_service.generate_embeddings(chunks)
        
        # Prepare metadata for vector store
        chunk_ids = [str(uuid.uuid4()) for _ in range(len(chunks))] 
        metadata = [
            {
                "document_id": document_id,
                "chunk_index": idx,
                "chunk_text": chunks[idx],
                "filename": file.filename,
                "file_type": file_type
            }
            for idx in range(len(chunks))
        ]
        
        # Store in vector database
        logger.info("Storing vectors in vector database")
        await vector_store.upsert_vectors(embeddings, chunk_ids, metadata)
        
        # Store metadata in SQL database
        logger.info("Storing metadata in database")
        async with db_manager.get_session() as session:
            doc_metadata = await db_manager.save_document_metadata(
                session=session,
                document_id=document_id,
                filename=file.filename,
                file_type=file_type,
                file_size=len(file_content),
                chunking_strategy=chunking_strategy,
                total_chunks=len(chunks)
            )
            
            await db_manager.save_chunks(
                session=session,
                document_id=document_id,
                chunks=chunks
            )
        
        logger.info(f"Document {document_id} processed successfully")
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            file_type=file_type,
            file_size=len(file_content),
            chunking_strategy=chunking_strategy,
            total_chunks=len(chunks),
            upload_timestamp=doc_metadata.upload_timestamp,
            message="Document uploaded and processed successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )


@router.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "vector_db": settings.vector_db_type,
        "database": settings.db_type,
        "embedding_model": settings.embedding_model
    }

