from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "RAG Pipeline API"
    app_version: str = "1.0.0"
    debug: bool = False

    # Embedding
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # Vector Store
    vector_store_type: str = "chroma"  # "chroma" | "milvus"

    # Chroma (default)
    chroma_persist_directory: str = "./data/chroma_db"
    chroma_collection_name: str = "rag_collection"

    # Milvus / Zilliz Cloud
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "rag_collection"
    milvus_embedding_dim: int = 384  # must match embedding_model_name output dim

    # Text Splitter
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # LLM — common
    llm_type: str = "openai_compatible"  # "ollama" | "openai_compatible" | "openai"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # Ollama (native /api/generate)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # OpenAI-compatible (vLLM, LM Studio, gpt-oss, etc.)
    openai_compatible_base_url: str = "http://192.168.0.163:8000"
    openai_compatible_model: str = "gpt-oss-20b"

    # OpenAI official (or Azure OpenAI)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_base_url: str = "https://api.openai.com/v1"

    # Retrieval
    retrieval_top_k: int = 5

    # OCR
    ocr_enabled: bool = True
    tesseract_lang: str = "eng"

    # Upload
    upload_directory: str = "./data/uploads"

    # Performance metrics
    metrics_enabled: bool = False  # set METRICS_ENABLED=true in .env to enable
    metrics_max_events: int = 10000  # max raw events kept in memory before oldest are dropped

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
