from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_demo.domain.models import DocumentChunk
from rag_demo.domain.ports import DocumentLoaderPort


class PdfDocumentLoader(DocumentLoaderPort):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load(self, file_path: Path) -> list[DocumentChunk]:
        reader = PdfReader(file_path)

        full_text = ""
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            full_text += f"\n{page_text}"

        raw_chunks = self._splitter.split_text(full_text)

        return [
            DocumentChunk(
                content=chunk,
                metadata={
                    "source": file_path.name,
                    "chunk_index": str(i),
                },
            )
            for i, chunk in enumerate(raw_chunks)
        ]
