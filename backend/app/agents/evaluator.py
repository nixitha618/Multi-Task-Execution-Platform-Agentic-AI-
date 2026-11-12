from typing import Dict, Any
from backend.app.agents.base import BaseAgent
from backend.app.models.schemas import EvaluationResult

class EvaluatorAgent(BaseAgent):
    """Agent 6: Evaluates step execution completeness, quality, and triggers self-correction retry if needed."""

    def __init__(self):
        super().__init__(
            name="EvaluatorAgent",
            description="Performs rigorous self-evaluation, consistency verification, and quality checking."
        )

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = state.get("user_prompt", "")
        task_spec = state.get("task_spec", {})
        execution_plan = state.get("execution_plan", {})
        step_results = state.get("step_results", {})
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)

        det_result = state.get("deterministic_memory_result")
        if det_result or task_spec.get("task_type") == "memory_query":
            return {
                "evaluation": {
                    "is_satisfactory": True,
                    "score": 1.0,
                    "feedback": "Conversation session memory retrieved successfully with 100% precision.",
                    "issues": [],
                    "needs_retry": False,
                    "retry_instructions": None
                },
                "needs_retry": False,
                "retry_instructions": None
            }

        system_prompt = (
            "You are the Evaluator / Critic Agent in an autonomous AI task execution system.\n"
            "Review the user's objective, the execution plan, and the actual results produced by the tools.\n"
            "Determine:\n"
            "1. is_satisfactory: bool (True if the request was answered accurately and completely)\n"
            "2. score: float between 0.0 and 1.0\n"
            "3. feedback: detailed assessment\n"
            "4. issues: list of specific shortcomings, errors, or omissions\n"
            "5. needs_retry: bool (True ONLY if critical information or execution failed AND retries remain)\n"
            "6. retry_instructions: clear instructions on how the planner should fix the plan"
        )

        compact_results = {}
        if isinstance(step_results, dict):
            for k, v in step_results.items():
                if isinstance(v, dict):
                    compact_results[k] = {
                        "tool_name": v.get("tool_name"),
                        "success": v.get("success"),
                        "output_preview": str(v.get("output", ""))[:800],
                        "citations": v.get("citations", [])[:3]
                    }
                else:
                    compact_results[k] = str(v)[:800]

        user_content = (
            f"User Goal: {prompt}\n"
            f"Expected Objective: {task_spec.get('objective')}\n"
            f"Execution Steps Results: {compact_results}\n"
            f"Current Retry Count: {retry_count} (Max: {max_retries})\n"
        )

        # If already at max retries, do not allow retry loop
        eval_result: EvaluationResult = await self.llm.ainvoke_structured(
            prompt=user_content,
            schema=EvaluationResult,
            system_prompt=system_prompt,
            fast=True
        )

        # Safety override on max retries
        if retry_count >= max_retries:
            eval_result.needs_retry = False

        return {
            "evaluation": eval_result.model_dump(),
            "needs_retry": eval_result.needs_retry,
            "retry_instructions": eval_result.retry_instructions if eval_result.needs_retry else None
        }

evaluator_agent = EvaluatorAgent()
