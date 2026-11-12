from typing import Dict, Any, List
from backend.app.agents.base import BaseAgent
from backend.app.models.schemas import FinalResponse

class ResponseAgent(BaseAgent):
    """Agent 7: Synthesizes final results, formats markdown, attaches artifacts, and prepares user presentation."""

    def __init__(self):
        super().__init__(
            name="ResponseAgent",
            description="Formats final response with markdown, tables, citations, and downloadable artifacts."
        )

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = state.get("user_prompt", "")
        task_spec = state.get("task_spec", {})
        step_results = state.get("step_results", {})
        evaluation = state.get("evaluation", {})
        artifacts = state.get("artifacts", [])

        det_result = state.get("deterministic_memory_result")
        if det_result and det_result.get("formatted_answer"):
            return {
                "final_response": det_result["formatted_answer"],
                "final_artifacts": artifacts,
                "status": "COMPLETED"
            }

        # Check if step results contains memory output for memory_query tasks
        if task_spec.get("task_type") == "memory_query" and isinstance(step_results, dict):
            for s_data in step_results.values():
                if isinstance(s_data, dict) and s_data.get("tool_name") == "memory" and s_data.get("output"):
                    return {
                        "final_response": s_data["output"],
                        "final_artifacts": artifacts,
                        "status": "COMPLETED"
                    }

        conv_ctx = state.get("conversation_context", "")
        memory_context = state.get("memory_context", "")

        system_prompt = (
            "You are the Response Agent in an autonomous agentic AI platform. "
            "Formulate an exceptional, executive-quality markdown final response addressing the user's initial request.\n\n"
            "CRITICAL PERSONALIZATION & FORMATTING DIRECTIVES (STRICTLY MANDATORY):\n"
            "1. If user preferences specify a summary style (e.g. 'Always give 5 short bullet points with key takeaways'), "
            "YOU MUST STRICTLY FOLLOW THAT FORMAT. For example, if asked for a summary, provide precisely 5 short bullet points with the key takeaways.\n"
            "2. If user profile facts (such as college, university, name, organization, department) are recorded in the memory context, "
            "incorporate and honor them faithfully whenever relevant to the user's inquiry or identity.\n"
            "3. If the user specifies any output style, tone, or language preference, strictly comply throughout.\n"
            "4. If the user asks about a specific section or topic (such as projects, skills, or experience), provide an "
            "in-depth, well-structured breakdown focusing specifically on that section from the retrieved step outputs.\n"
            "5. Include key findings, calculations, tables, references to any generated charts or documents, and conclusion.\n"
            "6. Always include source citations with chunk indices.\n"
            "7. If conversational context is provided, maintain coherence with previous turns when answering follow-up queries.\n"
            "8. Do not include meta-commentary about your internal prompts."
        )

        # Format step outputs clearly and bound size to keep within Groq rate limits
        steps_summary = []
        if isinstance(step_results, dict):
            for s_idx, s_data in sorted(step_results.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0):
                tool = s_data.get("tool_name", "direct")
                raw_out = str(s_data.get("output", ""))
                out = (raw_out[:2200] + "\n... [content truncated for summary]") if len(raw_out) > 2200 else raw_out
                cites = s_data.get("citations", [])
                steps_summary.append(
                    f"Step {s_idx} [Tool: {tool}]:\n"
                    f"Output:\n{out}\n"
                    f"Citations: {', '.join(cites) if cites else 'None'}"
                )
        formatted_steps = "\n\n".join(steps_summary) if steps_summary else str(step_results)[:4000]

        user_content = (
            f"User Prompt: {prompt}\n"
            f"Objective: {task_spec.get('objective')}\n"
            f"Executed Steps & Outputs:\n{formatted_steps}\n"
            f"Artifacts Generated: {artifacts}\n"
            f"Quality Evaluation: {evaluation.get('feedback', 'Satisfactory')}\n"
        )
        if conv_ctx:
            safe_conv = conv_ctx[-1500:] if len(conv_ctx) > 1500 else conv_ctx
            user_content += f"\nRecent Conversation History:\n{safe_conv}\n"
        if memory_context:
            safe_mem = memory_context[:2500] if len(memory_context) > 2500 else memory_context
            user_content += f"\n=== USER PROFILE & ACTIVE PERSONALIZATION DIRECTIVES (MANDATORY TO FOLLOW) ===\n{safe_mem}\n=================================================================================\n"

        response_text = await self.llm.ainvoke_text(
            prompt=user_content,
            system_prompt=system_prompt,
            fast=False
        )

        return {
            "final_response": response_text,
            "final_artifacts": artifacts,
            "status": "COMPLETED"
        }

response_agent = ResponseAgent()
