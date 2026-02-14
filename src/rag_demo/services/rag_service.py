import logging
from collections.abc import Iterator

from rag_demo.domain.models import DocumentChunk, QueryResult
from rag_demo.domain.ports import LLMPort, VectorStorePort

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the information provided below to answer the user's question.\n"
    "Rules:\n"
    "1. Answer in natural, complete sentences.\n"
    "2. If the answer is not in the provided information, simply say 'I don't know' without explanation.\n"
    "3. Do not mention 'the text', 'the context', or 'provided information' in your response.\n"
    "4. Do not make up facts or use outside knowledge.\n\n"
    "Information:\n{context}\n\n"
    "Question: {question}"
)


class RAGService:
    def __init__(
        self,
        llm: LLMPort,
        vector_store: VectorStorePort,
        top_k: int = 5,
    ) -> None:
        self._llm = llm
        self._store = vector_store
        self._top_k = top_k

    def _retrieve(self, question: str) -> tuple[list[DocumentChunk], str]:
        relevant_chunks = self._store.similarity_search(
            query=question, top_k=self._top_k
        )
        logger.info("Retrieved %d chunks", len(relevant_chunks))

        context_block = "\n\n---\n\n".join(
            chunk.content for chunk in relevant_chunks
        )
        prompt = _SYSTEM_PROMPT.format(context=context_block, question=question)
        return relevant_chunks, prompt

    def query(self, question: str) -> QueryResult:
        logger.info("RAG query: %s", question)

        relevant_chunks, prompt = self._retrieve(question)

        if not relevant_chunks:
            return QueryResult(
                answer="I don't have any relevant context to answer that question.",
                sources=[],
            )

        answer = self._llm.generate(prompt)
        logger.info("Generated answer (%d chars)", len(answer))

        return QueryResult(answer=answer, sources=relevant_chunks)

    def query_stream(self, question: str) -> Iterator[str]:
        logger.info("RAG stream query: %s", question)

        relevant_chunks, prompt = self._retrieve(question)

        if not relevant_chunks:
            yield "I don't have any relevant context to answer that question."
            return

        yield from self._llm.generate_stream(prompt)
