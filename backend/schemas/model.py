from datetime import datetime
from typing import Optional, Literal
from sqlalchemy import Column, String, DateTime, Integer, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field



Base = declarative_base()


class DocumentMetadata(Base):
    """SQLAlchemy model for document metadata."""
    
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    chunking_strategy = Column(String, nullable=False)
    total_chunks = Column(Integer, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON, default={})


class ChunkMetadata(Base):
    """SQLAlchemy model for chunk metadata."""
    
    __tablename__ = "chunks"
    
    id = Column(String, primary_key=True, index=True)
    document_id = Column(String, nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_size = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# Pydantic Schemas
class DocumentUploadResponse(BaseModel):
    """Response schema for document upload."""
    
    document_id: str
    filename: str
    file_type: str
    file_size: int
    chunking_strategy: str
    total_chunks: int
    upload_timestamp: datetime
    message: str


class ChunkingStrategy(BaseModel):
    """Chunking strategy configuration."""
    
    strategy: Literal["fixed", "semantic"] = Field(
        default="fixed",
        description="Chunking strategy: 'fixed' for fixed-size chunks, 'semantic' for sentence-based"
    )
    chunk_size: Optional[int] = Field(
        default=None,
        description="Size of each chunk (characters). Uses config default if not provided."
    )
    chunk_overlap: Optional[int] = Field(
        default=None,
        description="Overlap between chunks. Uses config default if not provided."
    )
