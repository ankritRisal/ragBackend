from backend.db.vector import create_vector_store
from backend.services.embedding import EmbeddingService
from backend.db.database import DatabaseManager
from backend.db.chatMemory import RedisMemoryManager
from backend.services.text_processing import TextExtractor, ChunkingService
from backend.llmModels.llm import create_llm_service
from backend.llmModels.rag import CustomRAGService
from config.settings import Settings
from typing import Optional

settings = Settings()

memory_manager = RedisMemoryManager(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    password=settings.redis_password,
    ttl=settings.redis_ttl
)
db_manager = DatabaseManager(settings.database_url)
vector_store = create_vector_store()
embedding_service = EmbeddingService(settings.embedding_model)
llm_service = create_llm_service()
text_extractor = TextExtractor()

        
# rag_service: Optional[CustomRAGService] = None

# # Global services
# settings = Settings()
# vector_store = create_vector_store()
# embedding_service = EmbeddingService(settings.embedding_model)
# memory_manager = RedisMemoryManager(
#     host=settings.redis_host,
#     port=settings.redis_port,
#     db=settings.redis_db,
#     password=settings.redis_password,
#     ttl=settings.redis_ttl
# )
# llm_service = create_llm_service()
rag_service: Optional[CustomRAGService] = CustomRAGService(
    vector_store=vector_store,
    embedding_service=embedding_service,
    llm_service=llm_service,
    memory_manager=memory_manager,
    top_k=settings.retrieval_top_k,
    similarity_threshold=settings.similarity_threshold,
    max_context_length=settings.max_context_length
)