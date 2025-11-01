from typing import List
import PyPDF2
from io import BytesIO
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter


class TextExtractor:
    """Extract text from various file formats."""
    def __init__(self):
        pass

    @staticmethod
    async def extract_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF file."""
        try:
            pdf_file = BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_parts: List[str] = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            return "\n\n".join(text_parts)
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    @staticmethod
    async def extract_from_txt(file_content: bytes) -> str:
        """Extract text from TXT file."""
        try:
            return file_content.decode("utf-8")
        except UnicodeDecodeError:
            # Try with different encoding
            return file_content.decode("latin-1")


class ChunkingService:
    """Service for chunking text using different strategies."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_fixed(self, text: str) -> List[str]:
        """
        Fixed-size chunking using RecursiveCharacterTextSplitter.
        Splits on paragraphs, then sentences, then characters.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(text)
        return chunks
    
    def chunk_semantic(self, text: str) -> List[str]:
        """
        Semantic chunking using sentence transformers.
        Groups semantically similar sentences together.
        """
        splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=self.chunk_overlap,
            tokens_per_chunk=self.chunk_size // 4  # Approximate tokens
        )
        chunks = splitter.split_text(text)
        return chunks
    
    async def chunk_text(
        self,
        text: str,
        strategy: str = "fixed"
    ) -> List[str]:
        """Chunk text using specified strategy."""
        if strategy == "fixed":
            return self.chunk_fixed(text)
        elif strategy == "semantic":
            return self.chunk_semantic(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")