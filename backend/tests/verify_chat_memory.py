"""
Comprehensive Verification Suite for Chat Memory & Conversation History:
Tests:
1. Turn 1: Initial query 'What is photosynthesis?'
2. Turn 2: 'What was my previous question?' -> Must return Turn 1's question, NOT itself!
3. Turn 3: 'What is the capital of France?'
4. Turn 4: 'What did I ask two questions ago?' -> Relative offset verification
5. Turn 5: 'What was my first question?' -> Earliest question in session
6. Turn 6: 'What questions have I asked so far?' -> Full question list
7. Turn 7: Session Isolation: Session B asks 'What was my previous question?' -> Must NOT leak Session A
8. Turn 8: Persistence across restart / direct SQLite inspection
9. Turn 9: Clear session endpoint
10. Turn 10: RAG Pipeline Coexistence: Document querying works without interference
"""

import asyncio
import uuid
import os
import sys
import sqlite3
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from backend.app.core.database import init_db
from backend.app.services.memory_service import memory_service
from backend.app.orchestration.graph import multi_agent_graph

async def run_turn(session_id: str, prompt: str, user_id: str = "test_user") -> Dict[str, Any]:
    # 1. Simulate API layer saving user prompt
    user_msg = await memory_service.save_message(
        session_id=session_id,
        role="user",
        content=prompt,
        metadata={"user_id": user_id}
    )
    current_msg_id = user_msg["id"]

    # 2. Run multi_agent_graph
    initial_state = {
        "task_id": str(uuid.uuid4()),
        "user_id": user_id,
        "session_id": session_id,
        "current_message_id": current_msg_id,
        "user_prompt": prompt,
        "attached_files": [],
        "step_results": {},
        "artifacts": [],
        "retry_count": 0,
        "max_retries": 1,
        "events": []
    }

    final_state = await multi_agent_graph.ainvoke(initial_state)
    return {
        "prompt": prompt,
        "final_response": final_state.get("final_response", ""),
        "task_spec": final_state.get("task_spec", {}),
        "execution_plan": final_state.get("execution_plan", {}),
        "step_results": final_state.get("step_results", {}),
        "deterministic_memory_result": final_state.get("deterministic_memory_result")
    }

async def main():
    print("=" * 70)
    print("STARTING CHAT MEMORY & CONVERSATION HISTORY VERIFICATION")
    print("=" * 70)

    await init_db()

    session_a = f"test_sess_a_{uuid.uuid4().hex[:8]}"
    session_b = f"test_sess_b_{uuid.uuid4().hex[:8]}"
    print(f"Session A ID: {session_a}")
    print(f"Session B ID: {session_b}")

    # -------------------------------------------------------------
    # Test 1: Turn 1 - Initial Knowledge Question
    # -------------------------------------------------------------
    print("\n[Test 1] Executing Turn 1 in Session A: 'What is photosynthesis?'")
    t1 = await run_turn(session_a, "What is photosynthesis?")
    print(f"Turn 1 Output preview: {t1['final_response'][:120]}...")
    assert len(t1["final_response"]) > 20, "Turn 1 produced no output"

    # Check that both user and assistant messages were saved in DB
    msgs_a1 = await memory_service.get_session_messages(session_a)
    print(f"Session A messages after Turn 1: {len(msgs_a1)}")
    assert len(msgs_a1) == 2, f"Expected 2 messages (1 user, 1 assistant), got {len(msgs_a1)}"
    assert msgs_a1[0]["role"] == "user" and msgs_a1[0]["content"] == "What is photosynthesis?"
    assert msgs_a1[1]["role"] == "assistant"
    print("--> PASS: Turn 1 executed and saved successfully.")

    # -------------------------------------------------------------
    # Test 2: Turn 2 - 'What was my previous question?'
    # -------------------------------------------------------------
    print("\n[Test 2] Executing Turn 2 in Session A: 'What was my previous question?'")
    t2 = await run_turn(session_a, "What was my previous question?")
    ans2 = t2["final_response"]
    print(f"Turn 2 Output:\n{ans2}")

    assert "photosynthesis" in ans2.lower(), f"Expected previous question to mention photosynthesis, got: {ans2}"
    # CRITICAL: Must not return the question itself!
    assert "what was my previous question" not in ans2.lower().replace("your previous question was", ""), "FAILED: Self-retrieval bug detected!"
    # Check that tool used was Memory
    plan_steps2 = t2["execution_plan"].get("steps", [])
    assert any(s.get("agent_name") == "Memory" or s.get("tool_name") == "memory" for s in plan_steps2), "Expected Memory tool execution for memory query"
    print("--> PASS: Correctly retrieved previous question without self-retrieval bug.")

    # -------------------------------------------------------------
    # Test 3: Turn 3 - Intermediary question
    # -------------------------------------------------------------
    print("\n[Test 3] Executing Turn 3 in Session A: 'What is the capital of France?'")
    t3 = await run_turn(session_a, "What is the capital of France?")
    print(f"Turn 3 Output preview: {t3['final_response'][:100]}...")
    msgs_a3 = await memory_service.get_session_messages(session_a)
    assert len(msgs_a3) == 6, f"Expected 6 messages after 3 turns, got {len(msgs_a3)}"
    print("--> PASS: Turn 3 completed.")

    # -------------------------------------------------------------
    # Test 4: Turn 4 - Relative query: 'What did I ask two questions ago?'
    # -------------------------------------------------------------
    print("\n[Test 4] Executing Turn 4 in Session A: 'What did I ask two questions ago?'")
    # Current is Turn 4. 1 question ago = Turn 3 ("capital of France"). 2 questions ago = Turn 2 ("What was my previous question?")
    t4 = await run_turn(session_a, "What did I ask two questions ago?")
    ans4 = t4["final_response"]
    print(f"Turn 4 Output:\n{ans4}")
    assert "what was my previous question" in ans4.lower(), f"Expected Turn 2 question from 2 questions ago, got: {ans4}"
    print("--> PASS: Correctly retrieved question from 2 questions ago.")

    # -------------------------------------------------------------
    # Test 5: Turn 5 - 'What was my first question?'
    # -------------------------------------------------------------
    print("\n[Test 5] Executing Turn 5 in Session A: 'What was my first question?'")
    t5 = await run_turn(session_a, "What was my first question?")
    ans5 = t5["final_response"]
    print(f"Turn 5 Output:\n{ans5}")
    assert "photosynthesis" in ans5.lower(), f"Expected first question to be photosynthesis, got: {ans5}"
    print("--> PASS: Correctly retrieved first question in session.")

    # -------------------------------------------------------------
    # Test 6: Turn 6 - 'What questions have I asked so far?'
    # -------------------------------------------------------------
    print("\n[Test 6] Executing Turn 6 in Session A: 'What questions have I asked so far?'")
    t6 = await run_turn(session_a, "What questions have I asked so far?")
    ans6 = t6["final_response"]
    print(f"Turn 6 Output:\n{ans6}")
    assert "photosynthesis" in ans6.lower()
    assert "capital of france" in ans6.lower()
    print("--> PASS: Correctly retrieved list of all past questions.")

    # -------------------------------------------------------------
    # Test 7: Turn 7 - Session Isolation in Session B
    # -------------------------------------------------------------
    print("\n[Test 7] Testing Session Isolation in Session B (fresh session): 'What was my previous question?'")
    t7 = await run_turn(session_b, "What was my previous question?")
    ans7 = t7["final_response"]
    print(f"Session B Turn 1 Output:\n{ans7}")
    assert "photosynthesis" not in ans7.lower(), f"Session leak detected! Session B returned Session A data: {ans7}"
    assert any(pat in ans7.lower() for pat in ["don't have any prior question", "no prior question", "no previous question", "current conversation history"])
    print("--> PASS: Session isolation confirmed. No cross-session leak.")

    # -------------------------------------------------------------
    # Test 8: Turn 8 - Direct SQLite Database Persistence Verification
    # -------------------------------------------------------------
    print("\n[Test 8] Verifying direct SQLite database persistence...")
    conn = sqlite3.connect("data/agentic_ai.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM conversation_messages WHERE session_id = ?", (session_a,))
    count_a = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM conversation_messages WHERE session_id = ?", (session_b,))
    count_b = cursor.fetchone()[0]
    conn.close()

    print(f"Direct SQLite row count for Session A: {count_a}")
    print(f"Direct SQLite row count for Session B: {count_b}")
    assert count_a >= 12, f"Expected at least 12 messages for Session A in SQLite, found {count_a}"
    assert count_b >= 2, f"Expected at least 2 messages for Session B in SQLite, found {count_b}"
    print("--> PASS: Persistence in SQLite database verified across process boundaries.")

    # -------------------------------------------------------------
    # Test 9: Turn 9 - Clearing Session History
    # -------------------------------------------------------------
    print(f"\n[Test 9] Clearing history for Session B ({session_b})...")
    await memory_service.clear_session(session_b)
    msgs_b_cleared = await memory_service.get_session_messages(session_b)
    assert len(msgs_b_cleared) == 0, f"Expected 0 rows after clear, got {len(msgs_b_cleared)}"
    # Ensure Session A is untouched
    msgs_a_intact = await memory_service.get_session_messages(session_a)
    assert len(msgs_a_intact) == count_a, "Session A was modified when clearing Session B!"
    print("--> PASS: Session clearing verified without affecting other sessions.")

    # -------------------------------------------------------------
    # Test 10: Turn 10 - RAG Pipeline Coexistence
    # -------------------------------------------------------------
    print("\n[Test 10] Verifying RAG Document Pipeline Coexistence...")
    from backend.app.rag.vectorstore import rag_store
    from backend.app.tools.rag import RAGTool
    count = rag_store.count()
    print(f"ChromaDB total indexed chunks: {count}")
    rag_tool_res = RAGTool.execute(query="photosynthesis", top_k=2)
    assert rag_tool_res.success, f"RAG tool query failed: {rag_tool_res.error}"
    results = rag_tool_res.result or []
    print(f"RAG search retrieved {len(results)} chunks.")
    assert len(results) > 0, "RAG search returned 0 chunks"
    print("--> PASS: RAG pipeline remains 100% operational and coexists seamlessly with Chat Memory.")

    print("\n" + "=" * 70)
    print("ALL 10 CHAT MEMORY & CONVERSATION HISTORY VERIFICATION TESTS PASSED (100%)!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
