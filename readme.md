# AlphaWave AI Assistant

Self-hosted Retrieval-Augmented Generation (RAG) AI assistant. Users authenticate, then ask questions in natural language against a private knowledge base scraped from `alphawave.hr`. All AI inference runs locally — no data leaves the machine.

---

## How It Works

1. User logs in via Supabase Auth (email + password)
2. User asks a question in the chat interface
3. The backend runs a **4-stage hybrid search** to find the most relevant document chunks:
   - Vector similarity (cosine distance via pgvector)
   - Keyword matching (ILIKE)
   - Reciprocal Rank Fusion (RRF) to merge both result lists
   - FlashRank cross-encoder reranker for final scoring
4. Top 5 chunks are injected as context into the LLM prompt
5. `qwen2.5:7b` generates a grounded answer, streamed token-by-token to the browser
6. Sources and the full interaction are logged

---

## Tech Stack

### Backend
| Component | Technology |
|-----------|------------|
| API Framework | FastAPI + Uvicorn |
| Language | Python 3 |
| LLM Orchestration | LangChain LCEL |
| LLM | Ollama — `qwen2.5:7b` |
| Embeddings | Ollama — `nomic-embed-text` (768-dim) |
| Reranker | FlashRank — `ms-marco-MiniLM-L-12-v2` |
| Database | PostgreSQL + pgvector |
| DB Driver | psycopg2 |
| Authentication | Supabase (JWT) |
| Streaming | Server-Sent Events (SSE) |
| Infrastructure | Docker Desktop |

### Frontend
| Component | Technology |
|-----------|------------|
| Framework | React + Vite |
| Auth Client | `@supabase/supabase-js` |
| Styling | Vanilla CSS |
| Communication | Fetch API (SSE streaming) |

---

## Prerequisites

- Python 3.10+
- Node.js and npm
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Ollama](https://ollama.com/)
- A [Supabase](https://supabase.com/) project (for auth and chat history)

---

## Setup

### 1. Environment Variables

Create a `.env` file in the project root:
```
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_ANON_KEY=<your-anon-key>
```

Create a `.env` file inside `frontend/`:
```
VITE_SUPABASE_URL=https://<your-project>.supabase.co
VITE_SUPABASE_ANON_KEY=<your-anon-key>
```

### 2. Database

Run PostgreSQL with pgvector in Docker:
```bash
docker run -d --name alphawave-db --restart unless-stopped \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=alphawave_ai \
  -p 5433:5432 \
  ankane/pgvector
```

Then enable the extension in pgAdmin (Tools → Query Tool):
```sql
CREATE EXTENSION vector;
```

### 3. AI Models

Pull the required models via Ollama:
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

### 4. Backend

```bash
python -m venv venv
.\venv\Scripts\activate
pip install fastapi uvicorn langchain-core langchain-ollama langchain-text-splitters flashrank psycopg2-binary supabase beautifulsoup4 requests python-dotenv
uvicorn app.api:app --reload
```

Backend runs on `http://127.0.0.1:8000`.

### 5. Ingest Knowledge Base

Crawl and scrape the website to populate the database (run once):
```bash
python -m app.scraper
```

### 6. Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173`.

---

## Startup Order (after initial setup)

| Step | What | Command |
|------|------|---------|
| 1 | Docker container | `docker start alphawave-db` |
| 2 | Ollama | Open Ollama app or `ollama serve` |
| 3 | Python backend | `uvicorn app.api:app --reload` |
| 4 | React frontend | `cd frontend && npm run dev` |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Blocking Q&A — returns full answer |
| `POST` | `/chat/stream` | Streaming Q&A via SSE |
| `GET` | `/history` | Fetch chat history for logged-in user |
| `POST` | `/history` | Save a message to chat history |
| `GET` | `/logs` | Fetch last 500 interaction logs |
| `GET` | `/health` | Health check |

All endpoints except `/health` require `Authorization: Bearer <token>`.

---

## Project Structure

```
├── app/
│   ├── api.py          # FastAPI REST API + Supabase auth
│   ├── rag.py          # RAG pipeline, LLM chain, SSE streaming
│   ├── database.py     # Hybrid search (vector + keyword + RRF + FlashRank)
│   ├── embeddings.py   # nomic-embed-text via LangChain + Ollama
│   ├── chunking.py     # RecursiveCharacterTextSplitter
│   ├── scraper.py      # BFS web crawler and data ingestion
│   └── logger.py       # JSONL interaction logger (rolling 500 entries)
├── frontend/
│   └── src/
│       ├── App.jsx          # Root — auth gate, idle timeout, routing
│       ├── Auth.jsx         # Login / registration (Supabase Auth)
│       ├── ChatWidget.jsx   # Floating popup chat widget
│       ├── FullChat.jsx     # Full-screen chat overlay
│       ├── Dashboard.jsx    # Live analytics (grouped by user + session)
│       └── supabaseClient.js
├── docs/
│   ├── project-documentation.md   # Full technical documentation
│   └── learning-notes.md
├── chat_logs.jsonl     # Rolling interaction log (auto-generated)
└── readme.md
```

---

## Documentation

Full technical documentation: [`docs/project-documentation.md`](docs/project-documentation.md)
