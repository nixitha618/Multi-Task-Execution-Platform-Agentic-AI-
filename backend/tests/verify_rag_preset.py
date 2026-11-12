import asyncio
import os
import sys

# Ensure backend root is in sys.path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath("."))

from backend.app.rag.vectorstore import rag_store
from backend.app.tools.rag import RAGTool
from backend.app.orchestration.graph import multi_agent_graph

async def run_task(prompt: str, attached_files: list = None):
    initial_state = {
        "task_id": f"test_rag_{os.urandom(3).hex()}",
        "user_id": "default_user",
        "user_prompt": prompt,
        "attached_files": attached_files or [],
        "chat_history": [],
        "memory_context": "",
        "execution_plan": {},
        "step_results": {},
        "artifacts": [],
        "events": [],
        "retry_count": 0,
        "max_retries": 2,
        "needs_retry": False,
        "retry_instructions": None,
        "final_output": None,
        "status": "PENDING"
    }
    return await multi_agent_graph.ainvoke(initial_state)

async def main():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE RAG DOCUMENT QUERY PRESET VERIFICATION")
    print("=" * 70)

    # 1. Check ChromaDB collection status and ensure initial documents
    print("\n--- 1. Vector Store Status & Document Auto-Indexing ---")
    rag_store.ensure_initial_documents()
    count = rag_store.count()
    print(f"Total chunks indexed in ChromaDB: {count}")
    assert count > 0, "Vector store should have indexed chunks"

    # 2. Test Direct RAG Tool execution
    print("\n--- 2. Direct RAGTool Execution ---")
    tool_res = RAGTool.execute(query="Agentic AI autonomous multi-agent systems", top_k=2)
    print(f"RAGTool success: {tool_res.success}")
    print(f"Matches returned: {len(tool_res.result)}")
    assert tool_res.success is True
    assert len(tool_res.result) > 0
    print(f"Top citation: {tool_res.result[0]['source']} (chunk {tool_res.result[0]['page']})")

    # 3. End-to-End Multi-Agent Workflow: Preset 3 (Agentic AI)
    print("\n--- 3. End-to-End Workflow: Preset 3 (Agentic AI) ---")
    p3_prompt = "Search knowledge base for information on Agentic AI and summarize key concepts."
    result_p3 = await run_task(p3_prompt)
    
    print(f"Preset 3 Status: {result_p3.get('status')}")
    final_output_p3 = result_p3.get("final_response", "")
    print(f"Output Length: {len(final_output_p3)} characters")
    print("\nPreset 3 Output Preview:")
    print(final_output_p3[:500] + "...\n")

    assert "Agentic AI" in final_output_p3, "Response must cover Agentic AI"
    assert "knowledge_repository.txt" in final_output_p3, "Response must cite knowledge_repository.txt"
    assert "ChromaDB" in final_output_p3, "Response must mention vector store"
    print(">>> PRESET 3 (AGENTIC AI) PASSED 100% <<<")

    # 4. End-to-End Multi-Agent Workflow: Photosynthesis Query
    print("\n--- 4. End-to-End Workflow: Photosynthesis Domain Query ---")
    photo_prompt = "Search knowledge base for information on photosynthesis and summarize key concepts."
    result_photo = await run_task(photo_prompt)
    final_photo = result_photo.get("final_response", "")
    print(f"Photosynthesis Output Length: {len(final_photo)} characters")
    assert "Photosynthesis" in final_photo
    assert "knowledge_repository.txt" in final_photo
    print(">>> PHOTOSYNTHESIS RAG QUERY PASSED 100% <<<")

    # 5. End-to-End Multi-Agent Workflow: Arbitrary PDF Document Query (Nixitha Resume)
    print("\n--- 5. End-to-End Workflow: Custom PDF Document Query (Nixitha Resume) ---")
    resume_prompt = "Search knowledge base for information on Nixitha's resume and summarize key qualifications."
    result_resume = await run_task(resume_prompt)
    final_resume = result_resume.get("final_response", "")
    print(f"Resume Query Output Length: {len(final_resume)} characters")
    print("\nResume Output Preview:")
    print(final_resume[:400] + "...\n")
    assert "Nixitha" in final_resume or "Resume" in final_resume
    assert "Nixitha_Resume (1).pdf" in final_resume or "chunk" in final_resume
    print(">>> CUSTOM PDF RAG QUERY PASSED 100% <<<")

    # 6. End-to-End Multi-Agent Workflow: Targeted Projects Section Query
    print("\n--- 6. End-to-End Workflow: Targeted Projects Section Query ---")
    projects_prompt = "Search knowledge base for information on Nixitha's resume and explain the projects section."
    result_projects = await run_task(projects_prompt)
    final_projects = result_projects.get("final_response", "")
    print(f"Projects Query Output Length: {len(final_projects)} characters")
    print("\nProjects Output Preview:")
    print(final_projects[:400] + "...\n")
    for proj in ["AI Mental Burnout Detection System", "Language Translation System", "Zoom AI Agent", "Google Calendar AI Agent"]:
        assert proj.lower() in final_projects.lower(), f"Missing project in response: {proj}"
        print(f"  [PASS] Verified project: {proj}")
    assert "Nixitha_Resume (1).pdf" in final_projects
    print(">>> TARGETED PROJECTS SECTION RAG QUERY PASSED 100% <<<")

    print("\n" + "=" * 70)
    print("ALL RAG DOCUMENT QUERY PRESETS & CUSTOM DOCUMENT SEARCHES VERIFIED 100%!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
