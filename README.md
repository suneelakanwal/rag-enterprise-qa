# rag-enterprise-qa

Production-grade **Retrieval-Augmented Generation** for company knowledge bases.  
Upload PDF/TXT/Markdown documents, ask questions, get answers grounded strictly in those docs — with confidence scores and a "no answer found" fallback.

> Built to mirror real enterprise RAG systems: strict grounding, failure-mode handling, and a clean API surface.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT / USER                               │
└───────────────┬─────────────────────────────┬───────────────────────┘
                │  POST /ingest               │  POST /ask
                ▼                             ▼
┌───────────────────────────────────────────────────────────────────┐
│                        FastAPI  (api.py)                           │
└───────┬───────────────────────────────────────────┬───────────────┘
        │                                           │
        ▼                                           ▼
┌───────────────────┐                  ┌────────────────────────────┐
│   ingest.py        │                  │        chain.py            │
│                   │                  │                            │
│  1. Load docs     │                  │  1. Call retriever         │
│     PDF/TXT/MD    │                  │  2. Check score threshold  │
│  2. Chunk text    │                  │     → "no answer" fallback │
│     (800 tok,     │                  │  3. Build grounded prompt  │
│      150 overlap) │                  │  4. Call OpenAI LLM        │
│  3. Upsert →      │                  │  5. Parse answer + JSON    │
│     ChromaDB      │                  │  6. Blend confidence score │
└────────┬──────────┘                  └────────────┬───────────────┘
         │                                          │
         ▼                                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                      retriever.py                                 │
│                                                                   │
│  • OpenAI text-embedding-3-small  (or local MiniLM fallback)      │
│  • ChromaDB cosine similarity search                              │
│  • min_score filter → drops irrelevant chunks                     │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                   ChromaDB (persistent)                           │
│                   ./chroma_db/                                    │
└──────────────────────────────────────────────────────────────────┘
```

### Confidence scoring

The `confidence` field (0–1) is a **blended score**:

```
confidence = (llm_self_assessment × 0.6) + (avg_retrieval_score × 0.4)
```

| Range | Interpretation |
|-------|---------------|
| ≥ 0.70 | High — well-supported by documents |
| 0.40–0.70 | Moderate — review cited sources |
| < 0.40 | Low — likely out-of-scope |

### RAG failure modes handled

| Failure | Detection | Response |
|---------|-----------|----------|
| Empty collection | `collection.count() == 0` | `no_answer_found=true`, no LLM call |
| No relevant chunks | All scores < `min_score` (0.30) | `no_answer_found=true`, no LLM call |
| Weak top result | Best score < `NO_ANSWER_THRESHOLD` (0.40) | `no_answer_found=true`, no LLM call |
| LLM signals uncertainty | Output contains "don't have enough information" | `grounded=false`, penalised confidence |
| OpenAI API error | Exception caught | `error` field set, safe fallback returned |

---

## Quick start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/rag-enterprise-qa.git
cd rag-enterprise-qa

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the API

```bash
uvicorn app.api:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### 4. Ingest documents

```bash
# Using the API
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"data_dir": "./data/sample_docs", "reset": true}'

# Or directly
python -m app.ingest ./data/sample_docs --reset
```

### 5. Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many remote work days are employees allowed per week?"}'
```

**Example response:**

```json
{
  "question": "How many remote work days are employees allowed per week?",
  "answer": "Employees are permitted to work remotely up to 3 days per week. [Source: employee_handbook.md]",
  "confidence": 0.87,
  "grounded": true,
  "no_answer_found": false,
  "sources": ["employee_handbook.md"],
  "retrieval_scores": [0.91, 0.74, 0.61]
}
```

**Out-of-scope question:**

```json
{
  "question": "What is the stock price of Acme Corp?",
  "answer": "I don't have enough information in the provided documents to answer this question.",
  "confidence": 0.12,
  "grounded": false,
  "no_answer_found": true,
  "sources": [],
  "retrieval_scores": []
}
```

---

## Docker

```bash
docker build -t rag-enterprise-qa .

docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/chroma_db:/app/chroma_db \
  rag-enterprise-qa
```

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/stats` | Vector store stats |
| `POST` | `/ingest` | Load documents into ChromaDB |
| `POST` | `/ask` | Ask a grounded question |
| `DELETE` | `/collection` | Wipe the vector store |

Full interactive docs at `http://localhost:8000/docs`.

### POST /ask — request body

```json
{
  "question": "string (required)",
  "top_k": 5,
  "min_score": 0.30,
  "filter_source": "pricing_guide.txt",
  "include_chunks": false
}
```

---

## Project structure

```
rag-enterprise-qa/
├── app/
│   ├── ingest.py        # Document loading + chunking → ChromaDB
│   ├── retriever.py     # Embedding + vector search + score filtering
│   ├── chain.py         # LLM prompt + grounding + confidence scoring
│   └── api.py           # FastAPI endpoints
├── data/sample_docs/
│   ├── employee_handbook.md
│   ├── pricing_guide.txt
│   └── api_documentation.txt
├── notebooks/
│   └── demo.ipynb       # End-to-end walkthrough
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Tech stack

| Component | Tool |
|-----------|------|
| Embeddings | OpenAI `text-embedding-3-small` (local fallback: `all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (persistent, cosine similarity) |
| LLM | OpenAI `gpt-4o-mini` (configurable) |
| Orchestration | LangChain (loaders + splitter) |
| API | FastAPI + Uvicorn |
| Containerisation | Docker |

---

## Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model |
| `CHUNK_SIZE` | `800` | Tokens per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `RETRIEVER_TOP_K` | `5` | Chunks to retrieve |
| `MIN_RETRIEVAL_SCORE` | `0.30` | Minimum cosine similarity |
| `NO_ANSWER_THRESHOLD` | `0.40` | Score below which no LLM call is made |
| `CHROMA_PATH` | `./chroma_db` | Vector store persistence path |

---

## License

MIT
