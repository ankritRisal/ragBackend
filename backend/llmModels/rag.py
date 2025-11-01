from typing import Any, Dict, Optional
from backend.db.chatMemory import RedisMemoryManager
from backend.db.vector import VectorStore
from backend.llmModels.llm import LLMService
from backend.services.embedding import EmbeddingService
from config.settings import Settings 
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class CustomRAGService:
    """Custom Retrieval-Augmented Generation service."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
        memory_manager: RedisMemoryManager,
        settings: Settings = Settings(),
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        max_context_length: int = 3000
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.settings = settings
        self.memory_manager = memory_manager
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.max_context_length = max_context_length
    
    async def retrieve_relevant_chunks(
        self,
        query: str
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for query."""
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embedding(query)
        
        # Search vector store
        results = await self.vector_store.search(
            vector=query_embedding,
            top_k=self.top_k,
            filter=None
        )
        
        # Filter by similarity threshold
        relevant_chunks = [
            {
                "chunk_id": result["id"],
                "text": result["metadata"]["chunk_text"],
                "score": result["score"],
                "document_id": result["metadata"]["document_id"],
                "filename": result["metadata"]["filename"]
            }
            for result in results
            if result["score"] >= self.similarity_threshold
        ]
        
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
        return relevant_chunks
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks."""
        if not chunks:
            return ""
        
        context_parts = []
        total_length = 0
        
        for i, chunk in enumerate(chunks, 1):
            chunk_text = chunk["text"]
            
            # Check if adding this chunk exceeds max context length
            if total_length + len(chunk_text) > self.max_context_length:
                break
            
            context_parts.append(
                f"[Source {i} - {chunk['filename']}]\n{chunk_text}"
            )
            total_length += len(chunk_text)
        
        return "\n\n".join(context_parts)
    
    async def generate_response(
        self,
        session_id: str,
        query: str,
        use_rag: bool = True
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate response using RAG.
        
        Returns:
            Tuple of (response_text, source_chunks)
        """
        sources = []
        
        # Retrieve conversation history
        conversation_history = await self.memory_manager.get_context_window(
            session_id,
            max_messages=6  # Last 3 exchanges
        )
        
        # Build messages for LLM
        messages = []
        
        if use_rag:
            # Retrieve relevant chunks
            sources = await self.retrieve_relevant_chunks(query)
            
            if sources:
                # Build context from retrieved chunks
                context = self._build_context(sources)
                
                # System prompt with context
                system_prompt = f"""You are a helpful AI assistant. Use the following context from documents to answer the user's question. If the context doesn't contain relevant information, say so and provide a helpful response based on your general knowledge.

                Context from documents:
                {context}

                Guidelines:
                - Answer based on the context when possible
                - Be concise and accurate
                - If booking an interview, guide the user through the process
                - Be conversational and friendly
                """
            else:
                # No relevant context found
                system_prompt = """You are a helpful AI assistant. The knowledge base doesn't contain information relevant to this query. Provide a helpful response based on your general knowledge, or guide the user to ask questions related to the available documents."""
        else:
            # No RAG, just conversational
            system_prompt = """You are a helpful AI assistant. Answer the user's questions in a friendly and informative manner."""
        
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history if exists
        if conversation_history:
            messages.append({
                "role": "system",
                "content": f"Previous conversation:\n{conversation_history}"
            })
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        # Generate response
        response = await self.llm_service.generate_response(
            messages=messages,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.max_tokens
        )
        
        # Store in memory
        await self.memory_manager.add_message(session_id, "user", query)
        await self.memory_manager.add_message(session_id, "assistant", response)
        
        return response, sources
    
    async def detect_booking_intent(self, query: str) -> bool:
        """Detect if user wants to book an interview."""
        booking_keywords = [
            "book", "schedule", "appointment", "interview",
            "meeting", "reserve", "set up", "arrange"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in booking_keywords)
