from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from rag_demo.domain.models import DocumentChunk


class LLMPort(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...

    @abstractmethod
    def generate_stream(self, prompt: str) -> Iterator[str]: ...


class EmbeddingPort(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class VectorStorePort(ABC):
    @abstractmethod
    def add_documents(self, chunks: list[DocumentChunk]) -> None: ...

    @abstractmethod
    def has_source(self, source_name: str) -> bool: ...

    @abstractmethod
    def similarity_search(
        self, query: str, top_k: int = 5
    ) -> list[DocumentChunk]: ...


class DocumentLoaderPort(ABC):
    @abstractmethod
    def load(self, file_path: Path) -> list[DocumentChunk]: ...
