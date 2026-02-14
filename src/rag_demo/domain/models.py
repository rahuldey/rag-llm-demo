from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentChunk:
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryResult:
    answer: str
    sources: list[DocumentChunk] = field(default_factory=list)
