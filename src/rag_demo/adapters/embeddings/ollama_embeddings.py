from langchain_ollama import OllamaEmbeddings

from rag_demo.domain.ports import EmbeddingPort


class OllamaEmbeddingAdapter(EmbeddingPort):
    def __init__(self, base_url: str, model: str) -> None:
        self._embeddings = OllamaEmbeddings(base_url=base_url, model=model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)
