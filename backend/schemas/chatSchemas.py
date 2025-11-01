from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Literal
from datetime import datetime


class ChatMessage(BaseModel):
    """Single chat message."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    session_id: str = Field(
        ...,
        description="Unique session ID for conversation tracking"
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User message"
    )
    use_rag: bool = Field(
        default=True,
        description="Whether to use RAG for this query"
    )


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    session_id: str
    message: str
    sources: List[dict] = Field(
        default_factory=list,
        description="Retrieved document chunks used for response"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatHistoryResponse(BaseModel):
    """Response schema for chat history endpoint."""
    session_id: str
    history: List[ChatMessage] = Field(
        default_factory=list,
        description="List of chat messages in the session"
    )

class BookingRequest(BaseModel):
    """Request schema for interview booking."""
    session_id: str
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    preferred_date: str = Field(
        ...,
        description="Preferred date in YYYY-MM-DD format"
    )
    preferred_time: str = Field(
        ...,
        description="Preferred time in HH:MM format (24-hour)"
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Additional notes or requirements"
    )


class BookingResponse(BaseModel):
    """Response schema for booking confirmation."""
    booking_id: str
    session_id: str
    name: str
    email: str
    preferred_date: str
    preferred_time: str
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    message: str