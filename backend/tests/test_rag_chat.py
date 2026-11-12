import os
import pytest
import uuid
from httpx import AsyncClient, ASGITransport
from backend.app.main import app
from backend.app.rag.vectorstore import rag_store
from backend.app.rag.ingestion import ingestion_service

@pytest.mark.asyncio
async def test_conversational_rag_chat_flow(tmp_path):
    # Pre-index test knowledge document into ChromaDB
    doc_path = tmp_path / "rag_test_manual.txt"
    doc_path.write_text(
        "Agentic AI features 7 specialized agents: Task Understanding, Planner, Research, Execution, Memory, Evaluator, and Response Agent. "
        "It uses LangGraph for DAG orchestration. "
        "The Research Agent conducts semantic vector search against ChromaDB using SentenceTransformers all-MiniLM-L6-v2 embeddings. "
        "It extracts grounded citations and supports continuous multi-turn conversational queries.",
        encoding="utf-8"
    )
    test_doc = "rag_test_manual.txt"
    chunks = ingestion_service.load_file(str(doc_path), user_id="test_user_rag")
    rag_store.add_documents(chunks)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = f"test_rag_session_{uuid.uuid4().hex[:8]}"

        # --- Turn 1: Initial Question ---
        req1 = {
            "message": "What are the 7 agents in Agentic AI?",
            "session_id": session_id,
            "user_id": "test_user_rag",
            "document_scope": test_doc,
            "top_k": 3
        }
        res1 = await client.post("/api/v1/documents/chat", json=req1)
        assert res1.status_code == 200, f"Turn 1 failed: {res1.text}"
        data1 = res1.json()
        assert data1["session_id"] == session_id
        assert len(data1["answer"]) > 10
        assert any("rag_test_manual.txt" in c for c in data1["citations"])
        assert len(data1["sources"]) > 0

        # --- Turn 2: Sequential Follow-Up Question referencing prior answer ---
        req2 = {
            "message": "Can you elaborate on the Research Agent mentioned in your previous answer?",
            "session_id": session_id,
            "user_id": "test_user_rag",
            "document_scope": test_doc,
            "top_k": 3
        }
        res2 = await client.post("/api/v1/documents/chat", json=req2)
        assert res2.status_code == 200, f"Turn 2 failed: {res2.text}"
        data2 = res2.json()
        assert data2["session_id"] == session_id
        assert len(data2["answer"]) > 10
        # The follow-up answer should address the Research Agent / embeddings
        assert any(term in data2["answer"].lower() for term in ["research", "chromadb", "embeddings", "agent"])

        # --- Turn 3: Check Chat History Endpoint ---
        hist_res = await client.get(f"/api/v1/documents/chat/history/{session_id}")
        assert hist_res.status_code == 200
        hist_data = hist_res.json()
        assert hist_data["session_id"] == session_id
        assert hist_data["total_messages"] >= 4 # 2 user questions + 2 assistant answers
        roles = [m["role"] for m in hist_data["messages"]]
        assert "user" in roles
        assert "assistant" in roles

        # --- Turn 4: Clear Chat Session ---
        del_res = await client.delete(f"/api/v1/documents/chat/history/{session_id}")
        assert del_res.status_code == 200
        assert del_res.json()["cleared"] is True

        # Verify history is now empty
        hist_after = await client.get(f"/api/v1/documents/chat/history/{session_id}")
        assert hist_after.status_code == 200
        assert hist_after.json()["total_messages"] == 0
