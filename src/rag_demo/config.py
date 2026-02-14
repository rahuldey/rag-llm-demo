from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_embed_model: str = "nomic-embed-text"
    chroma_persist_dir: str = "./chroma_data"
    chunk_size: int = 1000 # increase for better performance at cost of slowness
    chunk_overlap: int = 200 # increase for better performance at cost of slowness 
    top_k: int = 4 # increase for better performance at cost of slowness
    docs_dir: str = "./docs"
    ingest_max_retries: int = 1

    # Reads .env files if present, else uses sensible defaults
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
