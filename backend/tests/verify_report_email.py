import asyncio
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath("."))
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_report_email")

async def test_email_tool_direct():
    logger.info("=== 1. Testing EmailTool directly with alias & extra kwargs ===")
    from backend.app.tools.email_tool import EmailTool
    from backend.app.core.config import settings

    # Canonical call
    res1 = EmailTool.execute(
        recipient="user@example.com",
        subject="Direct Test Subject",
        body="Direct test email body",
        attachments=[]
    )
    assert res1.success, f"EmailTool failed: {res1.error}"
    assert res1.result.get("status") == "delivered"
    assert res1.result.get("recipient") == "user@example.com"
    logger.info("Pass: Canonical EmailTool call succeeded.")

    # Alias call with extra kwargs that previously caused crash (file_path)
    res2 = EmailTool.execute(
        to="client@example.org",
        title="Weekly Summary",
        content="Here is your report content",
        file_path="./data/uploads/student_performance.csv",
        unexpected_arg="should_be_ignored"
    )
    assert res2.success, f"EmailTool with aliases failed: {res2.error}"
    assert res2.result.get("status") == "delivered"
    assert res2.result.get("recipient") == "client@example.org"
    assert res2.result.get("subject") == "Weekly Summary"
    # Verify student_performance.csv was added to attachments because it exists
    assert any("student_performance.csv" in a for a in res2.artifacts), "Expected CSV in artifacts/attachments"
    # Verify .eml artifact was generated
    assert any(a.endswith(".eml") for a in res2.artifacts), "Expected .eml artifact in output directory"
    logger.info("Pass: Alias & extra kwargs EmailTool call succeeded with artifacts.")

async def test_file_gen_direct():
    logger.info("=== 2. Testing FileGenTool directly with aliases & extra kwargs ===")
    from backend.app.tools.file_gen import FileGenTool

    # Alias call with extra kwargs (file_path, to, body)
    res = FileGenTool.execute(
        filename="test_verification_report.md",
        report_content="# Automated Test Verification\n\nAll systems nominal.",
        file_path="./data/uploads/student_performance.csv",
        to="user@example.com"
    )
    assert res.success, f"FileGenTool failed: {res.error}"
    assert os.path.exists(res.result.get("file_path")), "Generated file must exist on disk"
    with open(res.result.get("file_path"), "r", encoding="utf-8") as f:
        c = f.read()
    assert "All systems nominal" in c
    logger.info(f"Pass: FileGenTool generated {res.result.get('filename')} successfully.")

async def test_tool_registry_defensive():
    logger.info("=== 3. Testing ToolRegistry defensive keyword filtering ===")
    from backend.app.tools.registry import tool_registry

    # Pass unexpected keyword arguments to ALL tools
    tools_to_test = [
        ("calculator", {"expression": "25 * 4 + 10", "file_path": "/bad/path", "to": "nobody"}),
        ("search", {"query": "python asyncio", "file_path": "/bad/path", "extra": 123}),
        ("file_gen", {"filename": "test_reg.md", "content": "hello", "file_path": "/bad/path"}),
        ("email", {"to": "reg@example.com", "subject": "Hi", "body": "test", "file_path": "/bad/path"}),
        ("rag", {"query": "test query", "file_path": "/bad/path", "unknown": True}),
    ]

    for t_name, t_kwargs in tools_to_test:
        r = await tool_registry.aexecute(t_name, **t_kwargs)
        assert r.success, f"Tool '{t_name}' failed under unexpected kwargs: {r.error}"
        logger.info(f"Pass: ToolRegistry executed '{t_name}' without parameter mismatch.")

async def test_e2e_student_performance_report_email():
    logger.info("=== 4. Testing End-to-End: student_performance.csv summary + email ===")
    from backend.app.orchestration.graph import multi_agent_graph

    prompt = "Generate a markdown summary of the student_performance.csv dataset and send an email notification to user@example.com"
    attached_files = ["./data/uploads/student_performance.csv"]

    initial_state = {
        "task_id": "test_student_perf_email",
        "user_id": "default_user",
        "session_id": "test_student_perf_email",
        "user_prompt": prompt,
        "attached_files": attached_files,
        "chat_history": [],
        "memory_context": "",
        "execution_plan": {},
        "step_results": {},
        "artifacts": [],
        "events": [],
        "retry_count": 0,
        "max_retries": 2,
    }

    final_state = await multi_agent_graph.ainvoke(initial_state)

    step_results = final_state.get("step_results", {})
    logger.info(f"Execution finished with {len(step_results)} steps.")

    for s_idx, s_data in step_results.items():
        t_name = s_data.get("tool_name")
        success = s_data.get("success")
        logger.info(f"Step {s_idx} [{t_name}]: success={success}, error={s_data.get('error')}")
        assert success, f"Step {s_idx} ({t_name}) failed: {s_data.get('error')}"

    # Verify artifacts
    artifacts = final_state.get("artifacts", [])
    logger.info(f"Generated artifacts: {artifacts}")
    assert any(a.endswith(".md") for a in artifacts), "Must generate a markdown report artifact"
    assert any(a.endswith(".eml") for a in artifacts), "Must generate an email delivery (.eml) artifact"

    # Verify final response
    resp = final_state.get("final_response", "")
    logger.info(f"Final response preview:\n{resp[:300]}...")
    assert "user@example.com" in resp or "Delivered" in resp or "delivered" in resp.lower(), "Response should confirm email delivery"
    logger.info("Pass: student_performance.csv report & email workflow succeeded completely!")

async def test_e2e_preset_report_email():
    logger.info("=== 5. Testing End-to-End: Preset 4 (Project deliverables report + email) ===")
    from backend.app.orchestration.graph import multi_agent_graph

    prompt = "Generate a markdown summary report of current project deliverables and send an email notification to user@example.com"

    initial_state = {
        "task_id": "test_preset_report_email",
        "user_id": "default_user",
        "session_id": "test_preset_report_email",
        "user_prompt": prompt,
        "attached_files": [],
        "chat_history": [],
        "memory_context": "",
        "execution_plan": {},
        "step_results": {},
        "artifacts": [],
        "events": [],
        "retry_count": 0,
        "max_retries": 2,
    }

    final_state = await multi_agent_graph.ainvoke(initial_state)

    step_results = final_state.get("step_results", {})
    for s_idx, s_data in step_results.items():
        t_name = s_data.get("tool_name")
        success = s_data.get("success")
        logger.info(f"Step {s_idx} [{t_name}]: success={success}, error={s_data.get('error')}")
        assert success, f"Step {s_idx} ({t_name}) failed: {s_data.get('error')}"

    artifacts = final_state.get("artifacts", [])
    logger.info(f"Preset 4 artifacts: {artifacts}")
    assert any(a.endswith(".md") for a in artifacts), "Must generate a markdown report artifact"

    resp = final_state.get("final_response", "")
    assert "user@example.com" in resp or "Delivered" in resp or "delivered" in resp.lower()
    logger.info("Pass: Preset 4 workflow completed successfully!")

async def main():
    await test_email_tool_direct()
    await test_file_gen_direct()
    await test_tool_registry_defensive()
    await test_e2e_student_performance_report_email()
    await test_e2e_preset_report_email()
    logger.info("========================================")
    logger.info("ALL REPORT & EMAIL TESTS PASSED WITH 100% SUCCESS!")
    logger.info("========================================")

if __name__ == "__main__":
    asyncio.run(main())
