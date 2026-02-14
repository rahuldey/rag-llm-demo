import asyncio
import logging
import os
import signal
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from rag_demo.adapters.document.pdf_loader import PdfDocumentLoader
from rag_demo.adapters.embeddings.ollama_embeddings import OllamaEmbeddingAdapter
from rag_demo.adapters.llm.ollama_llm import OllamaLLMAdapter
from rag_demo.adapters.vectorstore.chroma_store import ChromaVectorStoreAdapter
from rag_demo.api.routes import _get_app_status, _get_rag_service, router
from rag_demo.config import Settings
from rag_demo.services.ingest_service import IngestService
from rag_demo.services.rag_service import RAGService

logger = logging.getLogger(__name__)

_app_status: str = "ingesting"


def _set_status(new_status: str) -> None:
    global _app_status
    _app_status = new_status


async def _run_ingestion(ingest_service: IngestService, docs_dir: Path, max_retries: int) -> None:
    attempts = 0

    while attempts <= max_retries:
        try:
            total = await ingest_service.ingest_directory(docs_dir)
            logger.info("Startup ingestion finished: %d total chunks", total)
            _set_status("ready")
            return
        except Exception:
            attempts += 1
            if attempts <= max_retries:
                logger.exception(
                    "Ingestion attempt %d failed — retrying …", attempts
                )
            else:
                logger.exception(
                    "Ingestion failed after %d attempts — shutting down",
                    attempts,
                )
                _set_status("failed")
                os.kill(os.getpid(), signal.SIGTERM)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    docs_dir = Path(settings.docs_dir)

    embedding_adapter = OllamaEmbeddingAdapter(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
    )

    llm_adapter = OllamaLLMAdapter(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )

    vector_store = ChromaVectorStoreAdapter(
        persist_directory=settings.chroma_persist_dir,
        embedding_port=embedding_adapter,
    )

    document_loader = PdfDocumentLoader(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    ingest_service = IngestService(
        document_loader=document_loader,
        vector_store=vector_store,
    )
    
    rag_service = RAGService(
        llm=llm_adapter,
        vector_store=vector_store,
        top_k=settings.top_k,
    )

    app.dependency_overrides[_get_rag_service] = lambda: rag_service
    app.dependency_overrides[_get_app_status] = lambda: _app_status

    _set_status("ingesting")
    ingestion_task = asyncio.create_task(
        _run_ingestion(ingest_service, docs_dir, settings.ingest_max_retries)
    )

    yield

    if not ingestion_task.done():
        ingestion_task.cancel()


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    app = FastAPI(
        title="RAG LLM Demo",
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.state.settings = settings
    app.include_router(router)
    return app
