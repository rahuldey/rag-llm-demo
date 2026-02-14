import os

from langchain_chroma import Chroma

from rag_demo.domain.models import DocumentChunk
from rag_demo.domain.ports import EmbeddingPort, VectorStorePort


class ChromaVectorStoreAdapter(VectorStorePort):
    def __init__(
        self,
        persist_directory: str,
        embedding_port: EmbeddingPort,
        collection_name: str = "rag_documents",
    ) -> None:
        os.makedirs(persist_directory, exist_ok=True)
        
        self._store = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embedding_port,
        )

    def has_source(self, source_name: str) -> bool:
        results = self._store.get(where={"source": source_name}, limit=1)
        return len(results["ids"]) > 0

    def add_documents(self, chunks: list[DocumentChunk]) -> None:
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        self._store.add_texts(texts=texts, metadatas=metadatas)

    def similarity_search(
        self, query: str, top_k: int = 5
    ) -> list[DocumentChunk]:
        results = self._store.similarity_search(query, k=top_k)
        return [
            DocumentChunk(
                content=doc.page_content,
                metadata={k: str(v) for k, v in doc.metadata.items()},
            )
            for doc in results
        ]
