import asyncio
import logging
from pathlib import Path

from rag_demo.domain.ports import DocumentLoaderPort, VectorStorePort

logger = logging.getLogger(__name__)


class IngestService:
    def __init__(
        self,
        document_loader: DocumentLoaderPort,
        vector_store: VectorStorePort,
    ) -> None:
        self._loader = document_loader
        self._store = vector_store

    def ingest_file(self, file_path: Path) -> int:
        if self._store.has_source(file_path.name):
            logger.info("Skipping %s — already ingested", file_path.name)
            return 0

        logger.info("Ingesting file: %s", file_path.name)
        chunks = self._loader.load(file_path)
        if not chunks:
            logger.warning("No content extracted from %s", file_path.name)
            return 0
        self._store.add_documents(chunks)
        logger.info("Stored %d chunks from %s", len(chunks), file_path.name)
        return len(chunks)

    async def ingest_directory(self, directory: Path) -> int:
        if not directory.is_dir():
            raise FileNotFoundError(f"Docs directory not found: {directory}")

        supported_extensions = {".pdf", ".txt", ".ppt", ".pptx"}
        files_to_ingest = [
            f for f in sorted(directory.iterdir())
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        if not files_to_ingest:
            logger.warning("No supported files (pdf, txt, ppt, pptx) found in %s", directory)
            return 0

        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(None, self.ingest_file, file_path)
            for file_path in files_to_ingest
        ]
        results = await asyncio.gather(*tasks)
        total_chunks = sum(results)

        logger.info(
            "Ingestion complete: %d chunks from %d files",
            total_chunks,
            len(files_to_ingest),
        )
        return total_chunks
