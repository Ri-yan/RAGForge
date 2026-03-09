# RAG Pipeline API

A generic, production-grade Retrieval-Augmented Generation pipeline exposed via a REST API. Built with **SOLID principles** and fully open-source components.

## Architecture

```
src/
├── config/              # Settings & environment config
├── core/
│   ├── interfaces/      # Abstract base classes (SOLID contracts)
│   │   ├── document_loader.py
│   │   ├── text_splitter.py
│   │   ├── embedder.py
│   │   ├── vector_store.py
│   │   └── llm.py
│   ├── models.py        # Domain models (Document, Chunk)
│   └── pipeline.py      # RAG pipeline orchestrator
├── infrastructure/      # Concrete implementations
│   ├── document_loaders/  (PDF, TXT, OCR PDF, Images — extensible via factory)
│   ├── text_splitters/    (Recursive character splitter)
│   ├── embedders/         (Sentence-Transformers)
│   ├── vector_stores/     (ChromaDB)
│   └── llms/              (Ollama)
├── services/            # Business logic layer
│   ├── ingestion_service.py
│   └── query_service.py
├── api/                 # REST API layer
│   ├── routes/          # FastAPI routers
│   ├── schemas/         # Pydantic request/response models
│   └── dependencies.py  # Dependency injection wiring
└── main.py              # Application entry point
```

## SOLID Principles

| Principle | Implementation |
|-----------|---------------|
| **Single Responsibility** | Each class has one job — loaders load, splitters split, etc. |
| **Open/Closed** | New document loaders, embedders, or LLMs can be added without changing existing code |
| **Liskov Substitution** | All implementations are interchangeable through their abstract interfaces |
| **Interface Segregation** | Separate ABCs for each pipeline stage — no fat interfaces |
| **Dependency Inversion** | Pipeline and services depend on abstractions, not concrete classes |

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) running locally (for LLM inference)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on the system (for OCR support)
  - **Windows:** Download installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
  - **Linux:** `sudo apt install tesseract-ocr`
  - **macOS:** `brew install tesseract`
- [Poppler](https://poppler.freedesktop.org/) (required by `pdf2image` for scanned PDF OCR)
  - **Windows:** Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases) and add to PATH
  - **Linux:** `sudo apt install poppler-utils`
  - **macOS:** `brew install poppler`

### Setup

```bash
# Clone & enter project
cd RAG

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Copy env config
cp .env.example .env

# Pull an Ollama model
ollama pull llama3

# Run the server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/ingest/upload` | Upload & ingest documents |
| `POST` | `/api/v1/query/` | Ask a question |

### Usage Examples

**Ingest a document:**
```bash
curl -X POST http://localhost:8000/api/v1/ingest/upload \
  -F "files=@document.pdf"
```

**Ingest a scanned PDF or image (OCR):**
```bash
curl -X POST http://localhost:8000/api/v1/ingest/upload \
  -F "files=@scanned_document.pdf"

curl -X POST http://localhost:8000/api/v1/ingest/upload \
  -F "files=@photo_of_page.png"
```

**Ask a question:**
```bash
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

### Interactive Docs

Visit `http://localhost:8000/docs` for the Swagger UI.

## Docker

```bash
docker build -t rag-pipeline .
docker run -p 8000:8000 --env-file .env rag-pipeline
```

## Extending the Pipeline

**Add a new document loader** (e.g., DOCX):

1. Create `src/infrastructure/document_loaders/docx_loader.py` implementing `DocumentLoader`
2. Register it in `factory.py`:
   ```python
   factory.register_loader(DocxLoader())
   ```

**Swap the vector store** (e.g., to Qdrant):

1. Create `src/infrastructure/vector_stores/qdrant_store.py` implementing `VectorStore`
2. Update `dependencies.py` to return the new implementation

**Swap the LLM** (e.g., to vLLM or HuggingFace):

1. Create a new class implementing `LLMProvider`
2. Update `dependencies.py`

## Testing

```bash
pytest tests/ -v
```
