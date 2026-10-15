# 🛡️ RecallGuard

> **Temporal Memory Reliability Firewall for Autonomous AI Agents**

RecallGuard is a local-first memory reliability firewall designed to prevent autonomous AI agents from hallucinating outdated assumptions, acting on expired temporal context, and suffering from contradictory multi-session instructions.

---

## 📌 The Problem: Stale & Contradictory Memory Decay

When autonomous agents are equipped with long-term episodic memory, two catastrophic failure modes emerge:

1. **Stale Memory (Temporal Obsolescence):** Facts valid at timestamp $T_0$ (e.g., "backend runs on Node.js 20" or "release scheduled for Q2") expire over time. Standard vector and keyword search retrieve these expired memories as if they were current ground truth.
2. **Contradictory Memory (Logical Inconsistency):** Multi-session interactions record opposing mandates (e.g., "Node.js 20" in Session 1 vs "migrated to Go 1.22" in Session 4). Without temporal reconciliation, agents sample conflicting claims arbitrarily.

RecallGuard addresses this by decomposing memories into atomic claims tagged with explicit temporal bounds (`valid_from`, `valid_until`), lifecycle statuses (`active`, `superseded`, `needs_review`), verbatim source quotes, and confidence scores. It acts as an inline firewall before prompt construction.

---

## 🏗️ Project Architecture

```
recallguard/
├── api.py                      # Feature 05: FastAPI Local Agent Integration API
├── app.py                      # Feature 01–04: Interactive Streamlit Dashboard & Reliability Lab
├── core/
│   ├── __init__.py             # Core package interface & exports
│   ├── schemas.py              # Pydantic schemas (MemoryClaim, ClaimStatus, OllamaConfig)
│   ├── database.py             # Transactional SQLite manager with non-destructive migrations
│   ├── llm.py                  # Local Ollama client (qwen3.5:0.8b default)
│   ├── memory_service.py       # Claim extraction, quote verification & lifecycle management
│   ├── conflict_detector.py    # Deterministic temporal conflict resolution engine
│   ├── retrieval_service.py    # Feature 04: Evidence-grounded retrieval & cited answer synthesis
│   └── evaluator.py            # Feature 04: Reliability evaluation & synthetic benchmarking
├── data/
│   ├── demo_cases.json         # Realistic temporal memory failure scenarios
│   └── recallguard.db          # Local SQLite storage (gitignored)
├── tests/
│   ├── test_database.py        # Database initialization & CRUD tests
│   ├── test_memory_service.py  # Extraction, quote verification & persistence tests
│   ├── test_conflict_resolution.py # Temporal conflict detector & state transitions
│   ├── test_retrieval.py       # Evidence-grounded retrieval, keyword ranking & answer tests
│   ├── test_evaluator.py       # Reliability Lab benchmarks & metric consistency tests
│   └── test_api.py             # Feature 05: FastAPI endpoint & integration tests
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git protection rules
└── README.md                   # Documentation & guide
```

---

## 🔒 Security & Local-First Principles

- **Zero Cloud APIs:** All inference runs on your own machine using local [Ollama](https://ollama.ai) with `qwen3.5:0.8b`. No OpenAI, Anthropic, or Gemini API keys required.
- **Localhost Only:** The Agent API binds strictly to `127.0.0.1` by default and is never exposed publicly.
- **No Vector DB Bloat:** Uses transactional SQLite with deterministic indexing and verified verbatim source quotes.
- **Auditable & Non-Destructive:** Superseded and conflicting claims are preserved with lifecycle audit trails—never quietly deleted.

---

## ⚙️ Prerequisites & Environment

- **Python:** 3.10+ (tested with Python in `.venv`)
- **Ollama:** Installed and running locally
  ```bash
  # Check if Ollama is running
  ollama list

  # Pull the required lightweight model if not already present
  ollama pull qwen3.5:0.8b
  ```

---

## 🚀 Quickstart

### 1. Activate Environment & Install Dependencies
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the Local Agent API (FastAPI)
```bash
# Launch using the built-in runner (binds to 127.0.0.1:8000)
python api.py

# Or launch directly with uvicorn
uvicorn api:app --host 127.0.0.1 --port 8000
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive OpenAPI / Swagger Docs: `http://127.0.0.1:8000/docs`
- ReDoc Docs: `http://127.0.0.1:8000/redoc`

### 3. Start the Interactive Dashboard (Streamlit)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to explore:
- **Tab 1: Memory Explorer & AI Extraction:** Extract atomic claims from conversation text with verified source quotes.
- **Tab 2: Temporal Conflict Resolution:** Review conflicting claims with side-by-side evidence and audit trails.
- **Tab 3: Ask RecallGuard:** Test evidence-grounded retrieval and cited answers interactively.
- **Tab 4: Reliability Lab:** Benchmark RecallGuard vs naive baselines on synthetic datasets and audit live store health.

---

## 🔌 Local Agent Integration API (Feature 05)

The REST API allows any local autonomous agent framework (e.g. LangChain, LlamaIndex, AutoGen, CrewAI, or custom Python agents) to query memory reliability services over standard HTTP.

### Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service metadata, version, and documentation links |
| `GET` | `/health` | Live availability check for API, SQLite database, and Ollama daemon |
| `POST` | `/v1/memories/retrieve` | Retrieve active, non-expired evidence claims relevant to a query |
| `POST` | `/v1/ask` | End-to-end evidence-grounded answer with source quote citations and safe abstention |

---

### Endpoint Reference & Examples

#### 1. System Health Audit — `GET /health`

Reports the real-time operational status of the API, the SQLite database store, and the local Ollama LLM service.

**Request:**
```bash
curl -X GET http://127.0.0.1:8000/health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "api": "online",
  "database": {
    "connected": true,
    "path": "data/recallguard.db",
    "total_claims": 12,
    "active_claims": 9,
    "error": null
  },
  "ollama": {
    "connected": true,
    "host": "http://localhost:11434",
    "model": "qwen3.5:0.8b",
    "model_available": true,
    "error": null
  },
  "timestamp": "2026-10-15T00:15:00.000000Z"
}
```

*Note: If SQLite is reachable but Ollama is offline, the status will be `"degraded"`, indicating retrieval remains fully functional while grounded generation is temporarily disabled.*

---

#### 2. Evidence Retrieval — `POST /v1/memories/retrieve`

Searches memory records for active, non-expired claims matching the query keywords. Automatically filters out superseded, expired, and unresolved conflicting claims.

**Parameters:**
- `query` *(string, required)*: The question or keywords to search for. Must be non-empty.
- `max_results` *(integer, optional, default: 10, min: 1, max: 50)*: Maximum number of claims to return.
- `min_confidence` *(float, optional, default: 0.0, min: 0.0, max: 1.0)*: Minimum confidence rating.
- `include_needs_review` *(boolean, optional, default: false)*: Set to `true` to expose unresolved conflict claims.

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8000/v1/memories/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is our backend service framework?",
    "max_results": 5,
    "min_confidence": 0.8
  }'
```

**Example Response (200 OK):**
```json
{
  "query": "What is our backend service framework?",
  "claims": [
    {
      "id": "e4590abc-1234-5678-90ab-cdef12345678",
      "subject": "backend_service",
      "predicate": "framework",
      "object_value": "Go 1.22",
      "confidence": 0.95,
      "source_quote": "We decided to migrate the backend service from Node.js 20 to Go 1.22.",
      "source_session_id": "session-12",
      "valid_from": null,
      "valid_until": null,
      "created_at": "2026-10-14T18:00:00Z",
      "status": "active",
      "is_active": true,
      "superseded_by": null,
      "supersedes_id": "a1b2c3d4-...",
      "metadata": {}
    }
  ],
  "total_retrieved": 1,
  "filtered_out_count": 1,
  "confidence": 0.95,
  "abstained": false,
  "abstention_reason": ""
}
```

---

#### 3. Grounded Q&A with Verbatim Proof — `POST /v1/ask`

Retrieves active evidence, constructs a grounded local prompt, and synthesizes an answer using local Ollama. The response cites evidence IDs (e.g. `[E1]`) and verbatim quotes. If the memory store contains insufficient evidence, RecallGuard safely abstains rather than hallucinating.

**Parameters:**
- `query` *(string, required)*: Question about stored memories.
- `max_results` *(integer, optional, default: 10)*: Max evidence claims to supply to the LLM.
- `min_confidence` *(float, optional, default: 0.0)*: Minimum confidence threshold.

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Is our backend still running Node.js 20?"
  }'
```

**Example Response (Evidence Found):**
```json
{
  "query": "Is our backend still running Node.js 20?",
  "answer": "No, our backend service is no longer running Node.js 20; it has been migrated to Go 1.22 [E1]. The memory record confirms: \"We decided to migrate the backend service from Node.js 20 to Go 1.22.\"",
  "abstained": false,
  "abstention_reason": "",
  "evidence_ids": ["e4590abc-1234-5678-90ab-cdef12345678"],
  "evidence_quotes": ["We decided to migrate the backend service from Node.js 20 to Go 1.22."],
  "retrieved_claims": [
    {
      "id": "e4590abc-1234-5678-90ab-cdef12345678",
      "subject": "backend_service",
      "predicate": "framework",
      "object_value": "Go 1.22",
      "confidence": 0.95,
      "source_quote": "We decided to migrate the backend service from Node.js 20 to Go 1.22.",
      "status": "active",
      "is_active": true
    }
  ],
  "confidence": 0.95,
  "filtered_out_count": 1
}
```

**Example Response (Safe Abstention when Evidence is Absent):**
```json
{
  "query": "What is the secret corporate recipe for espresso beans?",
  "answer": "",
  "abstained": true,
  "abstention_reason": "No active memory claims matched the query.",
  "evidence_ids": [],
  "evidence_quotes": [],
  "retrieved_claims": [],
  "confidence": 0.0,
  "filtered_out_count": 0
}
```

---

### Python Integration Snippet

Integrate RecallGuard into your autonomous agent script with just a few lines:

```python
import httpx

RECALLGUARD_URL = "http://127.0.0.1:8000"

def get_grounded_context(user_query: str) -> dict:
    """Query RecallGuard firewall before generating an LLM agent response."""
    with httpx.Client(base_url=RECALLGUARD_URL, timeout=10.0) as client:
        response = client.post("/v1/ask", json={"query": user_query})
        response.raise_for_status()
        data = response.json()

        if data["abstained"]:
            print(f"⚠️ RecallGuard abstained: {data['abstention_reason']}")
            return {"has_evidence": False, "answer": None}

        print(f"✅ Grounded answer: {data['answer']}")
        print(f"📎 Proof quotes: {data['evidence_quotes']}")
        return {"has_evidence": True, "answer": data["answer"]}

# Example usage
context = get_grounded_context("What programming language does the user prefer?")
```

---

## 🧪 Testing & Verification

The test suite covers unit tests, temporal conflict scenarios, retrieval rankings, evaluation benchmarks, and FastAPI HTTP endpoints without requiring a live Ollama server or internet access.

```bash
# Run all automated tests
pytest -v

# Run only the Agent API test suite
pytest tests/test_api.py -v

# Run only retrieval & evaluation tests
pytest tests/test_retrieval.py tests/test_evaluator.py -v
```

---

## 🧭 Milestone Roadmap

- [x] **Feature 01: Project Foundation**
  - Streamlit dashboard & problem explainer
  - Pydantic schema for `MemoryClaim`
  - Safe, repeatable SQLite schema initialization & persistence
  - Local Ollama client configuration (`qwen3.5:0.8b`)
  - Demo failure cases & foundation test suite
- [x] **Feature 02: Local AI Memory Extraction**
  - Local conversation analysis with Qwen 3.5 (0.8B)
  - Atomic (Subject, Predicate, Object) structured JSON extraction
  - Verbatim source quote preservation and verification firewall
  - Malformed and ungrounded claim quarantine
  - Interactive user review & selective SQLite persistence
- [x] **Feature 03: Temporal Memory Conflict Resolution**
  - Deterministic comparison against existing active claims
  - Detection of exact duplicates, explicit updates, and ambiguous conflicts
  - Evidence-based transition detection via verified source quotes
  - Logical lifecycle states: `active`, `superseded`, and `needs_review`
  - Atomic database supersession preserving full historical audit trails
  - Interactive Conflict Review Queue and Decision Sandbox in Streamlit
- [x] **Feature 04: Evidence-Grounded Retrieval and Reliability Evaluation**
  - Keyword-based ranking over active memory claims
  - Automatic filtering of expired and superseded records
  - Evidence-grounded prompt construction citing exact Evidence IDs
  - Grounded answer generation with verbatim source quote proof
  - Safe abstention mechanism when evidence is absent
  - Synthetic evaluation benchmark & Reliability Radar dashboard
- [x] **Feature 05: Local Agent Integration API**
  - Standalone FastAPI application (`api.py`) bound to `127.0.0.1`
  - `GET /health`: Real-time audit of API, SQLite database, and Ollama daemon
  - `POST /v1/memories/retrieve`: Filtered retrieval of active, valid evidence claims
  - `POST /v1/ask`: Evidence-grounded Q&A with citations, source quotes, and safe abstention
  - Comprehensive automated test suite with mock LLM and in-memory database
  - Full documentation, OpenAPI schemas, and Python integration examples
