# Agentic AI System for Autonomous Multi-Task Execution

An end-to-end autonomous multi-agent task execution platform built with **FastAPI**, **LangGraph**, **ChromaDB**, **Sentence-Transformers**, and **Groq LLM**.

Unlike a conventional conversational chatbot, this platform operates as an **autonomous workflow engine**. Users supply natural-language instructions, datasets, documents, or voice commands. The system decomposes requests, dynamically generates execution plans, coordinates 7 specialized AI agents, invokes integrated tools (Python data analysis, calculator, web search, file generation, email notifications), queries private knowledge with RAG, evaluates its own outputs with a self-correction loop, and streams progress live to an interactive dashboard.

---

## Architecture Overview

```
USER (Text / Voice / Files)
  ↓
WEB DASHBOARD (HTML5/CSS3/JS) → FASTAPI REST LAYER & SSE STREAM
  ↓
1. MEMORY AGENT (Context Retrieval)
  ↓
2. TASK UNDERSTANDING AGENT (Intent & Entity Extraction)
  ↓
3. PLANNER AGENT (Multi-Step DAG Formulation)
  ↓
┌─────────────────────────── WORKER AGENTS ──────────────────────────┐
│                                                                   │
│   4. EXECUTION AGENT           5. RESEARCH AGENT (RAG)           │
│   ├─ Python Data Analysis      ├─ Document Parsing (PDF, DOCX)   │
│   ├─ Calculator & Math         ├─ Sentence-Transformers          │
│   ├─ Web Search                └─ Persistent ChromaDB Store      │
│   ├─ File Generation (MD/CSV)                                    │
│   └─ Email / Notifications                                       │
└───────────────────────────────────────────────────────────────────┘
  ↓
6. EVALUATOR / CRITIC AGENT
  │
  ├─ [Quality Score < 80% / Flaws Found] ──> Replanning & Retry Loop
  │
  └─ [Quality Score Satisfactory] ──>
  ↓
7. RESPONSE AGENT (Executive Markdown Synthesis)
  ↓
MEMORY PERSISTENCE → DASHBOARD LIVE OUTPUT / VOICE TTS
```

---

## The 7 Specialized Agents

1. **Task Understanding Agent** (`TaskUnderstandingAgent`): Decomposes natural-language prompts into formal JSON specifications (intent, task type, entities, tool requirements, constraints, and complexity).
2. **Planner Agent** (`PlannerAgent`): Translates specifications into ordered execution plans or DAGs of atomic steps, specifying tool inputs and step dependencies. Also performs dynamic replanning when instructed by the Evaluator.
3. **Research Agent** (`ResearchAgent`): Queries the ChromaDB vector database using semantic similarity search to extract relevant passages from uploaded documents and synthesizes findings with source citations.
4. **Execution Agent** (`ExecutionAgent`): Safely executes tools (Calculator, Python/Pandas sandbox, Web Search, File Generator, Email).
5. **Memory Agent** (`MemoryAgent`): Retrieves user-specific historical context, rules, and preferences prior to planning, and persists new learnings upon task completion into SQLite.
6. **Evaluator / Critic Agent** (`EvaluatorAgent`): Evaluates result completeness, relevance, and correctness, scoring the output and triggering automated retry iterations if needed.
7. **Response Agent** (`ResponseAgent`): Assembles all intermediate step outputs and file artifacts into an executive-ready markdown response.

---

## Core Tool Ecosystem

- **Python & Data Analysis Sandbox**: Reads CSV/Excel/JSON files, computes summary statistics with Pandas/NumPy, and automatically generates visualizations (line charts, bar charts, histograms) with Matplotlib.
- **Math Calculator**: Safe mathematical evaluation engine supporting algebraic formulas and trigonometric operations without shell or arbitrary code vulnerabilities.
- **Web Search**: Real-time web queries using DuckDuckGo with resilient fallback.
- **File Generator**: Creates downloadable Markdown reports, CSV files, and text documents saved into `./data/outputs/`.
- **Email Dispatcher**: Sends report notifications via SMTP or built-in mock mode.
- **Voice & Speech**: Built-in Speech-to-Text (STT) voice dictation and Text-to-Speech (TTS) readout.

---

## Project Directory Structure

```
Agentic_AI/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI entrypoint, static file mounts & lifespan
│   │   ├── core/
│   │   │   ├── config.py               # Pydantic settings & directories
│   │   │   ├── database.py             # SQLAlchemy async engine & session
│   │   │   └── llm.py                  # Groq API factory & structured output engine
│   │   ├── models/
│   │   │   ├── db_models.py            # SQLite schema (Task, Step, Memory, Log, Job)
│   │   │   └── schemas.py              # Pydantic schemas for tasks, plans, and events
│   │   ├── agents/
│   │   │   ├── base.py                 # Base agent abstract class
│   │   │   ├── task_understanding.py   # Agent 1
│   │   │   ├── planner.py              # Agent 2
│   │   │   ├── research.py             # Agent 3 (RAG)
│   │   │   ├── execution.py            # Agent 4 (Tools)
│   │   │   ├── memory_agent.py         # Agent 5 (Context & Memory)
│   │   │   ├── evaluator.py            # Agent 6 (Critic & Retry)
│   │   │   └── response.py             # Agent 7 (Final Output)
│   │   ├── tools/
│   │   │   ├── registry.py             # Unified tool dispatcher
│   │   │   ├── calculator.py           # Safe math evaluation
│   │   │   ├── search.py               # Web search
│   │   │   ├── data_analysis.py        # Pandas/NumPy/Matplotlib runner
│   │   │   ├── file_gen.py             # File exporter
│   │   │   └── email_tool.py           # Email sender
│   │   ├── rag/
│   │   │   ├── ingestion.py            # PDF/DOCX/TXT/CSV loader & text chunker
│   │   │   └── vectorstore.py          # Persistent ChromaDB + all-MiniLM-L6-v2
│   │   ├── orchestration/
│   │   │   ├── state.py                # LangGraph AgentState TypedDict
│   │   │   └── graph.py                # Multi-Agent StateGraph workflow DAG
│   │   ├── services/
│   │   │   ├── memory_service.py       # User memory management
│   │   │   └── scheduler.py            # APScheduler automated cron workflows
│   │   └── api/
│   │       ├── router.py               # API route aggregation
│   │       ├── endpoints_tasks.py      # Tasks & SSE live streaming
│   │       ├── endpoints_rag.py        # Document upload & knowledge retrieval
│   │       ├── endpoints_memory.py     # Memory CRUD
│   │       └── endpoints_schedule.py   # Automation & cron schedules
│   └── tests/
│       ├── test_agents.py
│       ├── test_tools.py
│       ├── test_rag.py
│       ├── test_orchestration.py
│       └── test_api.py
├── frontend/
│   ├── index.html                      # Interactive single-page dashboard
│   ├── css/
│   │   └── styles.css                  # Modern dark glassmorphic styling
│   └── js/
│       └── app.js                      # SSE client, agent visualizer, UI controls
├── data/
│   ├── chroma_db/                      # Persistent vector store
│   ├── uploads/                        # Uploaded user files
│   └── outputs/                        # Generated charts and reports
├── requirements.txt
├── pytest.ini
├── .env.example
└── README.md
```

---

## Quick Start & Installation

### 1. Prerequisites
- Python 3.11 (installed and accessible via `py -3.11`)

### 2. Environment Setup
The project already contains an initialized virtual environment (`venv`). To activate it in PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Configure API Key (Optional)
Copy `.env.example` to `.env` and set your Groq API key:
```env
GROQ_API_KEY="your-groq-api-key-here"
DEFAULT_MODEL="llama-3.3-70b-versatile"
FAST_MODEL="llama-3.1-8b-instant"
```
*(Note: If `GROQ_API_KEY` is not set, the platform will automatically run in intelligent local mock/rule-based mode, allowing full functionality and automated testing without internet or API dependencies).*

### 4. Run the Full Test Suite
Execute the 13 automated tests across tools, RAG, multi-agent orchestration, and API routes:
```powershell
.\venv\Scripts\pytest backend\tests\
```

### 5. Start the Application Server
Run the FastAPI application with Uvicorn:
```powershell
.\venv\Scripts\uvicorn backend.app.main:app --reload --port 8000
```

### 6. Access the System
- **Interactive Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative Redoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
#   M u l t i - T a s k - E x e c u t i o n - P l a t f o r m - A g e n t i c - A I -  
 