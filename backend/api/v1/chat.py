from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from backend.schemas.chatSchemas import ChatRequest, ChatResponse, ChatHistoryResponse, BookingRequest, BookingResponse
from backend.db.database import BookingInfo

from backend.services.services import memory_manager, rag_service, db_manager

import uuid
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Handle conversational chat with RAG support.
    
    - Maintains conversation history via session_id
    - Retrieves relevant context from vector store
    - Generates contextual responses
    - Supports both RAG and non-RAG queries
    """
    try:
        logger.info(f"Chat request - Session: {request.session_id}")
        
        # Generate response using RAG
        response_text, sources = await rag_service.generate_response(
            session_id=request.session_id,
            query=request.message,
            use_rag=request.use_rag
        )
        
        # Format sources for response
        formatted_sources = [
            {
                "chunk_id": src["chunk_id"],
                "filename": src["filename"],
                "relevance_score": round(src["score"], 3),
                "preview": src["text"][:200] + "..." if len(src["text"]) > 200 else src["text"]
            }
            for src in sources
        ]
        
        return ChatResponse(
            session_id=request.session_id,
            message=response_text,
            sources=formatted_sources
        )
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )


@router.post("/bookings", response_model=BookingResponse)
async def create_booking(request: BookingRequest) -> BookingResponse:
    """
    Create an interview booking.
    
    - Validates booking details
    - Stores in database
    - Returns confirmation
    """
    try:
        logger.info(f"Booking request - Session: {request.session_id}, Email: {request.email}")
        
        # Generate booking ID
        booking_id = str(uuid.uuid4())
        
        # Create booking record
        booking = BookingInfo(
            id=booking_id,
            session_id=request.session_id,
            name=request.name,
            email=request.email,
            preferred_date=request.preferred_date,
            preferred_time=request.preferred_time,
            notes=request.notes,
            status="pending"
        )
        async with db_manager.session_factory() as db_session:
            db_session.add(booking)
            await db_session.commit()
            await db_session.refresh(booking)
        
        logger.info(f"Booking created: {booking_id}")
        
        return BookingResponse(
            booking_id=booking_id,
            session_id=request.session_id,
            name=request.name,
            email=request.email,
            preferred_date=request.preferred_date,
            preferred_time=request.preferred_time,
            status="pending",
            created_at=booking.created_at,
            message="Interview booking created successfully. You will receive a confirmation email shortly."
        )
    
    except Exception as e:
        logger.error(f"Booking error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create booking: {str(e)}"
        )


# @router.get("/chat/history/{session_id}")
@router.get("/chat_history", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: str,
    limit: int = 20,
) -> ChatHistoryResponse:
    """Get conversation history for a session."""
    try:
        history = await memory_manager.get_history(session_id, limit=limit)
        return {
            "session_id": session_id,
            "message_count": len(history),
            "messages": history
        }
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(e)}"
        )


@router.delete("/chat/history/{session_id}" ,response_model= None)
async def clear_chat_history(
    session_id: str,
) -> dict:
    """Clear conversation history for a session."""
    try:
        await memory_manager.clear_history(session_id)
        return {
            "message": f"Chat history cleared for session {session_id}"
        }
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear history: {str(e)}"
        )

@router.get("/health", tags=["system"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for all core services.
    Verifies DB, Redis, Vector Store, and LLM readiness.
    """
    from datetime import datetime
    import asyncio

    health_status = {
        "timestamp": datetime.utcnow().isoformat(),
        "status": "healthy",
        "components": {}
    }

    try:
        # Check Database
        async with db_manager.session_factory() as session:
            await session.execute("SELECT 1")
        health_status["components"]["database"] = "Connected"
    except Exception as e:
        health_status["components"]["database"] = f"Error: {str(e)}"
        health_status["status"] = "degraded"

    try:
        # Check Redis Memory Manager
        test_key = "health_check_test"
        await memory_manager.set_message(test_key, {"test": "ok"})
        await memory_manager.clear_history(test_key)
        health_status["components"]["redis_memory"] = "Connected"
    except Exception as e:
        health_status["components"]["redis_memory"] = f"Error: {str(e)}"
        health_status["status"] = "degraded"

    try:
        # Check Vector Store (if it has async or sync test)
        if hasattr(rag_service.vector_store, "client"):
            health_status["components"]["vector_db"] = "Initialized"
        else:
            health_status["components"]["vector_db"] = "Unknown state"
    except Exception as e:
        health_status["components"]["vector_db"] = f"Error: {str(e)}"
        health_status["status"] = "degraded"

    try:
        # Check LLM service
        if rag_service.llm_service is not None:
            health_status["components"]["llm_service"] = "Ready"
        else:
            health_status["components"]["llm_service"] = "Not initialized"
    except Exception as e:
        health_status["components"]["llm_service"] = f"Error: {str(e)}"
        health_status["status"] = "degraded"

    try:
        # Embedding Model
        health_status["components"]["embedding_model"] = (
            f"✅ {rag_service.embedding_service.model_name}"
        )
    except Exception as e:
        health_status["components"]["embedding_model"] = f"Error: {str(e)}"

    return health_status