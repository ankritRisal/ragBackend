import json
from typing import List, Optional
import redis.asyncio as redis
from datetime import datetime


class RedisMemoryManager:
    """Manages conversation history using Redis."""
    
    def __init__(
        self,
        host: str,
        port: int,
        db: int = 0,
        password: Optional[str] = None,
        ttl: int = 3600
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.ttl = ttl
        self.client: Optional[redis.Redis] = None
    
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        self.client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password if self.password else None,
            decode_responses=True
        )
        # Test connection
        await self.client.ping()
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()
    
    def _get_key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"chat_session:{session_id}"
    
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> None:
        """Add a message to conversation history."""
        key = self._get_key(session_id)
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add to Redis list
        await self.client.rpush(key, json.dumps(message))
        
        # Set expiration
        await self.client.expire(key, self.ttl)
    
    async def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[dict]:
        """Retrieve conversation history."""
        key = self._get_key(session_id)
        
        if self.client is None:
            raise RuntimeError("Redis client not initialized. Call initialize() first.")

        # Get all messages or limited number
        if limit:
            messages = await self.client.lrange(key, -limit, -1)
        else:
            messages = await self.client.lrange(key, 0, -1)
        
        return [json.loads(msg) for msg in messages]
    
    async def clear_history(self, session_id: str) -> None:
        """Clear conversation history for session."""
        key = self._get_key(session_id)
        await self.client.delete(key)
    
    async def session_exists(self, session_id: str) -> bool:
        """Check if session exists."""
        key = self._get_key(session_id)
        return await self.client.exists(key) > 0
    
    async def get_context_window(
        self,
        session_id: str,
        max_messages: int = 10
    ) -> str:
        """Get recent conversation context as formatted string."""
        history = await self.get_history(session_id, limit=max_messages)
        
        if not history:
            return ""
        
        context_parts = []
        for msg in history:
            role = msg["role"].title()
            content = msg["content"]
            context_parts.append(f"{role}: {content}")
        
        return "\n\n".join(context_parts)
