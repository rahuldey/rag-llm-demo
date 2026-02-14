from pathlib import Path
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from rag_demo.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from rag_demo.services.rag_service import RAGService

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _get_rag_service() -> RAGService:
    raise RuntimeError("RAGService not initialised")

def _get_app_status() -> str:
    return "unknown"

@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )

@router.get("/health", response_model=HealthResponse)
def health(status: str = Depends(_get_app_status)) -> HealthResponse:
    return HealthResponse(status=status)

@router.post("/query", response_model=QueryResponse)
def query(
    body: QueryRequest,
    rag_service: RAGService = Depends(_get_rag_service),
) -> QueryResponse:
    result = rag_service.query(body.question)
    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceChunk(
                content=chunk.content,
                source=chunk.metadata.get("source", ""),
                chunk_index=chunk.metadata.get("chunk_index", ""),
            )
            for chunk in result.sources
        ],
    )

@router.post("/query/stream")
def query_stream(
    body: QueryRequest,
    rag_service: RAGService = Depends(_get_rag_service),
):
    return StreamingResponse(
        rag_service.query_stream(body.question),
        media_type="text/event-stream",
    )
