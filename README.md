# 🛡️ RecallGuard

> **Temporal Memory Reliability Firewall for Autonomous AI Agents**

RecallGuard is a local-first memory reliability firewall designed to prevent autonomous AI agents from hallucinating outdated assumptions, acting on expired temporal context, and suffering from contradictory multi-session instructions.

---

## 📌 The Problem: Stale & Contradictory Memory Decay

When autonomous agents are given long-term episodic memory, two catastrophic failure modes emerge:

1. **Stale Memory (Temporal Obsolescence):** Facts that were valid at timestamp $T_0$ (e.g., "production uses Python 3.9" or "release scheduled for Q2") expire over time. Standard semantic search retrieves these expired memories as if they were current ground truth.
2. **Contradictory Memory (Logical Inconsistency):** Multi-session agent interactions record opposing mandates (e.g., "use PostgreSQL" in Session 1 vs "switched to SQLite" in Session 4). Without temporal reconciliation, the agent samples conflicting claims arbitrarily.

RecallGuard addresses this by decomposing memories into atomic claims with explicit temporal boundaries (`valid_from`, `valid_until`) and auditing candidate memories before prompt construction.

---

## 🏗️ Project Architecture (Feature 01: Project Foundation)

```
recallguard/
├── app.py                      # Professional Streamlit UI & Explorer
├── core/
│   ├── __init__.py             # Core package interface
│   ├── schemas.py              # Pydantic schemas (MemoryClaim, OllamaConfig)
│   ├── database.py             # Safe, repeatable SQLite database manager
│   ├── llm.py                  # Local Ollama client (qwen3.5:0.8b default)
│   ├── memory_service.py       # Memory persistence & lifecycle service
│   └── evaluator.py            # Reliability Evaluator foundation stub
├── data/
│   ├── demo_cases.json         # Realistic temporal memory failure scenarios
│   └── recallguard.db          # Local SQLite storage (gitignored)
├── tests/
│   ├── test_database.py        # Database initialization & CRUD tests
│   ├── test_memory_service.py  # Memory service & claim lifecycle tests
│   └── test_evaluator.py       # Reliability evaluator contract tests
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git protection rules
└── README.md                   # Documentation & guide
```

---

## ⚙️ Prerequisites & Environment

- **Python:** 3.10+ (Current: Python in `.venv`)
- **Ollama:** Installed locally with model `qwen3.5:0.8b`
  ```bash
  ollama list
  # If not yet downloaded:
  ollama pull qwen3.5:0.8b
  ```
- **Local-First Rule:** Zero paid cloud inference APIs (no OpenAI, no Gemini, no external vector cloud).

---

## 🚀 Quickstart

### 1. Activate Environment
```bash
source .venv/bin/activate
```

### 2. Verify / Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run the Test Suite
```bash
pytest -v
```

---

## 🧭 Milestone Roadmap

- [x] **Feature 01: Project Foundation**
  - Streamlit dashboard & problem explainer
  - Pydantic schema for `MemoryClaim`
  - Safe, repeatable SQLite initialization & storage
  - Local Ollama client configuration (`qwen3.5:0.8b`)
  - Demo failure cases & test suite
- [x] **Feature 02: Local AI Memory Extraction**
  - Local conversation analysis with Qwen 3.5 (0.8B)
  - Atomic (Subject, Predicate, Object) structured JSON extraction
  - Verbatim source quote preservation and validation firewall
  - Malformed and ungrounded claim quarantine
  - Interactive user review & selective SQLite persistence
- [ ] **Feature 03:** Temporal Conflict & Contradiction Resolution
- [ ] **Feature 04:** Temporal Memory Retrieval Engine
- [ ] **Feature 05:** Reliability Evaluation & Firewall Filtering

