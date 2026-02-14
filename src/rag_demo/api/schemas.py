from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        json_schema_extra={"examples": ["What is this document about?"]},
    )

class SourceChunk(BaseModel):
    content: str
    source: str = ""
    chunk_index: str = ""

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = []

class HealthResponse(BaseModel):
    status: str
