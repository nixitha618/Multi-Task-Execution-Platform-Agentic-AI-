import pytest
from backend.app.orchestration.graph import multi_agent_graph

@pytest.mark.asyncio
async def test_end_to_end_calculation():
    initial_state = {
        "task_id": "test-calc-task-1",
        "user_id": "test_user",
        "user_prompt": "Calculate (45 * 12) + (180 / 3)",
        "attached_files": [],
        "step_results": {},
        "artifacts": [],
        "retry_count": 0,
        "max_retries": 1,
        "events": []
    }

    result = await multi_agent_graph.ainvoke(initial_state)

    assert result["status"] == "COMPLETED"
    assert "task_spec" in result
    assert "execution_plan" in result
    assert len(result["step_results"]) > 0
    assert result["final_response"] is not None
    assert len(result["events"]) > 5

@pytest.mark.asyncio
async def test_end_to_end_data_analysis():
    initial_state = {
        "task_id": "test-data-task-2",
        "user_id": "test_user",
        "user_prompt": "Analyze sales dataset and generate monthly revenue plot",
        "attached_files": [],
        "step_results": {},
        "artifacts": [],
        "retry_count": 0,
        "max_retries": 1,
        "events": []
    }

    result = await multi_agent_graph.ainvoke(initial_state)

    assert result["status"] == "COMPLETED"
    assert result["execution_plan"]["estimated_steps"] >= 2
    assert len(result["artifacts"]) > 0 # Chart artifact produced!
    assert "final_response" in result
