
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from config.settings import Settings
from qdrant_client import QdrantClient


class VectorStore(ABC):
    """Abstract base class for vector store implementations."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store connection."""
        pass
    
    @abstractmethod
    async def upsert_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Upsert vectors with metadata."""
        pass


################
    @abstractmethod
    async def search(
            self,
            vector: List[float],
            top_k: int = 5,
            filter: Optional[Dict[str, Any]] = None
        ) -> List[Dict[str, Any]]:
            """Search for similar vectors."""
            pass
    
    @abstractmethod
    async def delete_by_document_id(self, document_id: str) -> None:
        """Delete all vectors for a document."""
        pass

class QdrantStore(VectorStore):
    """Qdrant vector store implementation."""
    
    def __init__(self, url: str, api_key: str, collection_name: str = "documents"):
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.client = QdrantClient(url=self.url)
        self.settings = Settings()
    
    async def initialize(self) -> None:
        """Initialize Qdrant connection."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        if self.api_key:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            self.client = QdrantClient(url=self.url)
        
        # Create collection if not exists
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.settings.embedding_dimension,
                    distance=Distance.COSINE
                )
            )
    
    async def upsert_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Upsert vectors to Qdrant."""
        from qdrant_client.models import PointStruct
        
        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=metadata[i]
            )
            for i in range(len(vectors))
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    async def delete_by_document_id(self, document_id: str) -> None:
        """Delete vectors by document_id."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        )

###############
    async def search(
        self,
        vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search Qdrant for similar vectors."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        search_filter = None
        if filter:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter.items()
            ]
            search_filter = Filter(must=conditions)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=top_k,
            query_filter=search_filter
        )
        
        return [
            {
                "id": result.id,
                "score": result.score,
                "metadata": result.payload
            }
            for result in results
        ]
def create_vector_store() -> VectorStore:
    """Factory function to create vector store based on configuration."""
    settings = Settings()
    if settings.vector_db_type == "qdrant":
        return QdrantStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
    # Add Weaviate and Milvus implementations similarly
    else:
        raise ValueError(f"Unsupported vector DB type: {settings.vector_db_type}")
    
