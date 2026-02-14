# RAG LLM Demo

A local Retrieval-Augmented Generation API. Drop PDFs into a folder, start the server, and ask questions — answers are grounded in your documents. No third-party APIs, everything runs on your machine.

## Prerequisites

1. **Install Ollama**

   **Linux / macOS:**

   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

   **Windows:**
   Download and run the installer from [ollama.com/download](https://ollama.com/download).

2. **Pull the models**

   ```bash
   ollama pull llama3.2            # chat model (~2 GB)
   ollama pull nomic-embed-text    # embedding model (~274 MB)
   ```

3. **Install Python dependencies**
   ```bash
   uv sync
   ```

## Running

1. Place your PDF files in the `docs/` directory.

2. Start the server:

   ```bash
   uv run python main.py
   ```

3. The server starts at `http://localhost:8000`. On startup it automatically:
   - Scans `docs/` for PDFs
   - Skips files that have already been ingested (checked via vector database metadata)
   - Extracts and chunks the text from new files
   - Embeds the chunks and stores them in the vector database
   - If ingestion fails, it retries once — if the retry also fails, the server shuts down

4. Open the **Swagger UI** at `http://localhost:8000/docs` to interact with the API.
   Alternatively, use **ReDoc** at `http://localhost:8000/redoc` for a read-only documentation view.

## Running with Docker

You can also run the entire stack (App + Ollama) using Docker Compose.

1. **Prerequisites**: Ensure you have Docker and Docker Compose installed.

2. **Prepare Documents**: Place your PDF files in the `docs/` directory.

3. **Start the Stack**:

   ```bash
   docker compose up --build
   ```

   **Note on First Run**: The first time you run this, it will take several minutes because it needs to download the LLM models (~2.3 GB) inside the container. You will see logs from the `ollama-model-puller` container indicating progress.

   Once the models are downloaded, the `rag-app` service will start automatically.

4. **Access the App**:
   - Web UI: `http://localhost:8000`
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

   The `ollama_data` volume persists the models, so subsequent restarts will be instant.

## Flows

### Startup Ingestion

```
Server starts
  → Scans docs/ for PDF files
  → For each file, checks if it has already been ingested (metadata lookup)
  → Skips already-ingested files
  → Extracts text from new PDFs
  → Splits text into overlapping chunks
  → Embeds each chunk into a vector (concurrently across files)
  → Stores vectors in the local vector database
  → Status moves from "ingesting" → "ready"
```

### Query

```
User sends POST /query with a question
  → Question is embedded into a vector
  → Vector database returns the top-k most similar chunks
  → Chunks are assembled into a context block
  → Context + question are sent to the LLM as a prompt
  → LLM generates an answer grounded in the context
  → Answer + source references are returned as JSON
```

### Query (Streaming)

```
User sends POST /query/stream with a question
  → Same retrieval flow as /query
  → LLM generates tokens one at a time
  → Tokens are streamed to the client via Server-Sent Events (SSE)
```

Test with curl:

```bash
curl -N -X POST "http://localhost:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "what does ISRO do"}'
```

### Health Check

```
GET /health
  → Returns {"status": "ingesting"} while startup ingestion is running
  → Returns {"status": "ready"} once ingestion is complete
  → Returns {"status": "failed"} if ingestion failed after retries
```

## API Endpoints

| Method | Path            | Description                                       |
| ------ | --------------- | ------------------------------------------------- |
| `GET`  | `/health`       | Application status and ingestion state            |
| `POST` | `/query`        | Ask a question, returns full JSON response        |
| `POST` | `/query/stream` | Ask a question, streams the answer token by token |

## Configuration

All settings are controlled via environment variables (or a `.env` file). See `.env.example` for the full list of knobs including model names, chunk sizes, and retrieval parameters.
