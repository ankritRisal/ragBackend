from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.future import select
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from backend.schemas.model import DocumentMetadata, ChunkMetadata, Base
from typing import List
from sqlalchemy import Column, String, DateTime, Text
from datetime import datetime


class DatabaseManager:
    """Async database manager for PostgreSQL."""
    
    def __init__(self, database_url: str):
        self.engine = create_async_engine(database_url, echo=False)
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def initialize(self) -> None:
        """Create database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("Database tables ensured/created.")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session."""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def save_document_metadata(
        self,
        session: AsyncSession,
        document_id: str,
        filename: str,
        file_type: str,
        file_size: int,
        chunking_strategy: str,
        total_chunks: int
    ) -> DocumentMetadata:
        """Save document metadata to database."""
        doc_metadata = DocumentMetadata(
            id=document_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            chunking_strategy=chunking_strategy,
            total_chunks=total_chunks
        )
        session.add(doc_metadata)
        await session.flush()
        return doc_metadata
    
    async def save_chunks(
        self,
        session: AsyncSession,
        document_id: str,
        chunks: List[str]
    ) -> None:
        """Save chunk metadata to database."""
        chunk_objects = [
            ChunkMetadata(
                id=f"{document_id}_chunk_{idx}",
                document_id=document_id,
                chunk_index=idx,
                chunk_text=chunk_text,
                chunk_size=len(chunk_text)
            )
            for idx, chunk_text in enumerate(chunks)
        ]
        session.add_all(chunk_objects)
        await session.flush()


####################### Additional Model for Bookings #########################
class BookingInfo(Base):
    """SQLAlchemy model for interview bookings."""
    
    __tablename__ = "bookings"
    
    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    preferred_date = Column(String, nullable=False)
    preferred_time = Column(String, nullable=False)
    notes = Column(Text, nullable=True)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
