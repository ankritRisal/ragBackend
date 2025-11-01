from config.settings import Settings

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

settings = Settings()

class LLMService(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate response from LLM."""
        pass


class OpenAIService(LLMService):
    """OpenAI GPT service."""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate response using OpenAI."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content


# class AnthropicService(LLMService):
#     """Anthropic Claude service."""
    
#     def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
#         from anthropic import AsyncAnthropic
#         self.client = AsyncAnthropic(api_key=api_key)
#         self.model = model
    
#     async def generate_response(
#         self,
#         messages: List[Dict[str, str]],
#         temperature: float = 0.7,
#         max_tokens: int = 1000
#     ) -> str:
#         """Generate response using Anthropic."""
#         # Extract system message if present
#         system_msg = None
#         chat_messages = []
        
#         for msg in messages:
#             if msg["role"] == "system":
#                 system_msg = msg["content"]
#             else:
#                 chat_messages.append(msg)
        
#         response = await self.client.messages.create(
#             model=self.model,
#             max_tokens=max_tokens,
#             temperature=temperature,
#             system=system_msg,
#             messages=chat_messages
#         )
        
#         return response.content[0].text


def create_llm_service() -> LLMService:
    """Factory function to create LLM service."""
    if settings.llm_provider == "openai":
        return OpenAIService(
            api_key=settings.openai_api_key,
            model=settings.llm_model
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
