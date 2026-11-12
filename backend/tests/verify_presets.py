import asyncio
import os
import sys

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath("."))

from backend.app.orchestration.graph import multi_agent_graph

async def test_workflow(name: str, prompt: str):
    print(f"=== Testing: {name} ===")
    print(f"Prompt: {prompt}")
    initial_state = {
        "task_id": f"test_{name.lower().replace(' ', '_')}",
        "user_id": "default_user",
        "user_prompt": prompt,
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

    result = await multi_agent_graph.ainvoke(initial_state)
    print(f"Final Status: {result.get('status')}")
    print(f"Artifacts ({len(result.get('artifacts', []))}): {result.get('artifacts')}")
    output = result.get("final_response") or result.get("final_output", "")
    print("Output Preview:")
    print(output[:300] + ("..." if len(output) > 300 else ""))
    print("\n" + "="*50 + "\n")
    return result

async def main():
    # 1. Preset 1: Data Analysis & Chart
    r1 = await test_workflow(
        "Preset 1 - Data Analysis",
        "Analyze sales dataset, calculate monthly profit margins, generate a line chart, and produce an analytical report."
    )
    assert len(r1.get("artifacts", [])) >= 2, "Should have chart and report artifacts"
    out1 = r1.get("final_response") or r1.get("final_output", "")
    assert "profit margin" in out1.lower() or "sales" in out1.lower(), "Should contain profit margin table/summary"

    # 2. Preset 2: Math Calculation
    r2 = await test_workflow(
        "Preset 2 - Math Calculation",
        "Calculate ((14500 * 1.15) - 3200) / 4 and show step-by-step arithmetic."
    )
    out2 = r2.get("final_response") or r2.get("final_output", "")
    clean_out2 = out2.replace(r"\,", "").replace(",", "").replace(" ", "")
    assert "3368.75" in clean_out2, f"Should contain evaluated result 3368.75, got: {out2}"

    # 3. Preset 3: RAG Document Query
    r3 = await test_workflow(
        "Preset 3 - RAG Agentic AI",
        "Search knowledge base for information on Agentic AI and summarize key concepts."
    )
    out3 = r3.get("final_response") or r3.get("final_output", "")
    assert "agentic" in out3.lower() or "knowledge" in out3.lower() or "knowledge_repository.txt" in out3, "Should synthesize Agentic AI knowledge"

    # 4. Preset 4: Report & Email
    r4 = await test_workflow(
        "Preset 4 - Report & Email",
        "Generate a markdown summary report of current project deliverables and send an email notification to user@example.com"
    )
    assert any("deliverable" in str(a).lower() or ".md" in str(a).lower() for a in r4.get("artifacts", [])), "Should create deliverables report"
    out4 = r4.get("final_response") or r4.get("final_output", "")
    assert "user@example.com" in out4, "Should mention recipient user@example.com"

    # 5. Arbitrary Math Calculation (Generalization)
    r5 = await test_workflow(
        "Generalization - Custom Math",
        "Calculate sqrt(256) + 3^4 and show arithmetic"
    )
    out5 = r5.get("final_response") or r5.get("final_output", "")
    assert "97.0" in out5 or "97" in out5, f"Should calculate sqrt(256) + 3^4 = 97, got: {out5}"

    # 6. Arbitrary Email Recipient (Generalization)
    r6 = await test_workflow(
        "Generalization - Custom Email Recipient",
        "Compile project deliverables and email to lead.engineer@autonomous.ai"
    )
    out6 = r6.get("final_response") or r6.get("final_output", "")
    assert "lead.engineer@autonomous.ai" in out6, "Should email lead.engineer@autonomous.ai"

    # 7. Arbitrary Chart Type (Generalization)
    r7 = await test_workflow(
        "Generalization - Bar Chart",
        "Generate a bar chart of monthly revenue from dataset"
    )
    assert any(".png" in str(a) for a in r7.get("artifacts", [])), "Should generate chart artifact"

    print(">>> ALL PRESETS AND GENERALIZED TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    asyncio.run(main())