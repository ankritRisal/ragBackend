from fastapi import FastAPI
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))


from backend.services.services import memory_manager, vector_store, embedding_service, llm_service, rag_service, db_manager, settings 
from backend.llmModels.rag import CustomRAGService


import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Backend API",
    description="Document Ingestion and Conversational RAG with Interview Booking",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Backend API",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup."""
    logger.info("Initializing services...")
    await db_manager.initialize()
    await vector_store.initialize()
    # gloabalize rag_service
    global rag_service

    await memory_manager.initialize()
    logger.info("All services initialized successfully")

        # Test Redis connection immediately
    try:
        pong = await memory_manager.client.ping()
        if pong:
            logger.info("Redis client initialized successfully")
        else:
            logger.warning("Redis client ping returned False")
    except Exception as e:
        logger.error(f"Redis client failed to initialize: {str(e)}")
        raise RuntimeError("Redis initialization failed") from e
    
    rag_service = CustomRAGService(
        vector_store=vector_store,
        embedding_service=embedding_service,
        llm_service=llm_service,
        memory_manager=memory_manager,
        top_k=settings.retrieval_top_k,
        similarity_threshold=settings.similarity_threshold,
        max_context_length=settings.max_context_length
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down services...")
    await memory_manager.close()
    logger.info("Services shut down successfully")


# Include routers
from backend.api.v1 import ingestion, chat

app.include_router(ingestion.router)
app.include_router(chat.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
