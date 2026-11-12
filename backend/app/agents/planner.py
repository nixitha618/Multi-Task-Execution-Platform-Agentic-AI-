import os
import re
from typing import Dict, Any, List
from backend.app.agents.base import BaseAgent
from backend.app.models.schemas import ExecutionPlan, PlanStep
from backend.app.services.report_builder import report_builder
from backend.app.core.config import settings

class PlannerAgent(BaseAgent):
    """Agent 2: Converts high-level task understanding into an executable sequence/DAG of steps."""

    def __init__(self):
        super().__init__(
            name="PlannerAgent",
            description="Formulates multi-step execution plans and adapts to evaluator criticism."
        )

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        task_spec = state.get("task_spec", {})
        prompt = state.get("user_prompt", "")
        retry_instructions = state.get("retry_instructions", None)
        user_id = state.get("user_id", "default_user")
        memory_context = state.get("memory_context", "")

        from backend.app.services.memory_service import memory_service
        user_sig = await memory_service.get_user_preference(user_id, "signature")
        default_signoff = user_sig if user_sig else "Best regards,\nAutonomous Agentic AI Orchestrator"

        det_result = state.get("deterministic_memory_result")
        is_memory_task = (
            bool(det_result)
            or task_spec.get("task_type") == "memory_query"
            or "memory" in (task_spec.get("required_tools") or [])
        )
        if is_memory_task:
            return {
                "execution_plan": {
                    "task_objective": task_spec.get("objective") or f"Answer conversation memory query: {prompt}",
                    "steps": [
                        {
                            "step_index": 1,
                            "description": "Retrieve conversational record from session memory",
                            "agent_name": "Memory",
                            "tool_name": "memory",
                            "tool_input_template": {"query": prompt},
                            "depends_on": []
                        }
                    ],
                    "estimated_steps": 1
                },
                "status": "PLANNED"
            }

        # Resolve active attached document
        attached_files = list(state.get("attached_files") or [])
        from backend.app.services.report_builder import should_auto_associate_upload
        if not attached_files and should_auto_associate_upload(prompt, attached_files):
            if os.path.exists(settings.UPLOAD_DIR):
                recent_uploads = sorted(
                    [
                        f for f in os.listdir(settings.UPLOAD_DIR)
                        if not f.startswith(".") and os.path.isfile(os.path.join(settings.UPLOAD_DIR, f))
                    ],
                    key=lambda f: os.path.getmtime(os.path.join(settings.UPLOAD_DIR, f)),
                    reverse=True
                )
                if recent_uploads:
                    attached_files = [recent_uploads[0]]

        doc_ids = task_spec.get("document_ids") or []
        target_doc = attached_files[0] if attached_files else (doc_ids[0] if doc_ids else None)
        task_type = task_spec.get("task_type", "general")
        output_format = task_spec.get("output_format", "chat")
        send_email = task_spec.get("send_email", False)
        recipient = task_spec.get("recipient")

        is_doc_task = bool(
            target_doc and (
                task_type in [
                    "summary", "key_points", "extraction", "table", "report",
                    "detailed_summary", "explanation", "comparison", "question_answering", "rag_qa"
                ]
                or any(str(target_doc).lower().endswith(ext) for ext in [".pdf", ".docx", ".txt"])
                or "rag" in (task_spec.get("required_tools") or [])
            )
        )

        # If it is a document task and no replanning instructions are active, formulate optimal deterministic steps
        if is_doc_task and not retry_instructions:
            doc_name = os.path.basename(target_doc)
            objective = task_spec.get("objective") or f"Process {doc_name} for task: {task_type}"

            if send_email:
                unique_filename = report_builder.generate_unique_filename(task_type, doc_name=doc_name, prompt=prompt, ext="pdf")
                report_title = report_builder.generate_report_title(task_type, doc_name=doc_name, prompt=prompt)
                recip_email = recipient or "user@example.com"

                steps = [
                    PlanStep(
                        step_index=1,
                        description=f"Retrieve and synthesize {task_type.replace('_', ' ')} from {doc_name}",
                        agent_name="Research",
                        tool_name="rag",
                        tool_input_template={"query": prompt, "source": doc_name},
                        depends_on=[]
                    ),
                    PlanStep(
                        step_index=2,
                        description=f"Compile and generate formatted PDF document: {unique_filename}",
                        agent_name="Execution",
                        tool_name="file_gen",
                        tool_input_template={
                            "filename": unique_filename,
                            "file_type": "pdf",
                            "content": ""
                        },
                        depends_on=[1]
                    ),
                    PlanStep(
                        step_index=3,
                        description=f"Send email notification with generated PDF report to {recip_email}",
                        agent_name="Execution",
                        tool_name="email",
                        tool_input_template={
                            "recipient": recip_email,
                            "subject": report_title,
                            "body": f"Hello,\n\nPlease find the generated {task_type.replace('_', ' ')} report attached.\n\n{default_signoff}"
                        },
                        depends_on=[2]
                    )
                ]
                plan = ExecutionPlan(
                    objective=objective,
                    steps=steps,
                    estimated_steps=3
                )
                return {
                    "execution_plan": plan.model_dump(),
                    "status": "PLANNED"
                }

        # If it is a creative / topic generation task (no document) requiring email or PDF output
        if not is_doc_task and not retry_instructions and (send_email or output_format in ["pdf", "markdown"]):
            topic_title = report_builder.generate_report_title(task_type, doc_name=None, prompt=prompt)
            unique_filename = report_builder.generate_unique_filename(task_type, doc_name=None, prompt=prompt, ext="pdf")
            recip_email = recipient or "user@example.com"
            objective = task_spec.get("objective") or f"Generate content for '{topic_title}' and deliver via email"

            if send_email:
                steps = [
                    PlanStep(
                        step_index=1,
                        description=f"Synthesize content for '{topic_title}'",
                        agent_name="Research",
                        tool_name=None,
                        tool_input_template={"query": prompt},
                        depends_on=[]
                    ),
                    PlanStep(
                        step_index=2,
                        description=f"Compile and generate formatted PDF document: {unique_filename}",
                        agent_name="Execution",
                        tool_name="file_gen",
                        tool_input_template={
                            "filename": unique_filename,
                            "file_type": "pdf",
                            "content": ""
                        },
                        depends_on=[1]
                    ),
                    PlanStep(
                        step_index=3,
                        description=f"Send email notification with generated PDF document to {recip_email}",
                        agent_name="Execution",
                        tool_name="email",
                        tool_input_template={
                            "recipient": recip_email,
                            "subject": topic_title,
                            "body": f"Hello,\n\nPlease find the generated document for '{topic_title}' attached.\n\n{default_signoff}"
                        },
                        depends_on=[2]
                    )
                ]
                plan = ExecutionPlan(
                    objective=objective,
                    steps=steps,
                    estimated_steps=3
                )
                return {
                    "execution_plan": plan.model_dump(),
                    "status": "PLANNED"
                }
            elif output_format in ["pdf", "markdown"]:
                steps = [
                    PlanStep(
                        step_index=1,
                        description=f"Synthesize content for '{topic_title}'",
                        agent_name="Research",
                        tool_name=None,
                        tool_input_template={"query": prompt},
                        depends_on=[]
                    ),
                    PlanStep(
                        step_index=2,
                        description=f"Compile and generate formatted PDF document: {unique_filename}",
                        agent_name="Execution",
                        tool_name="file_gen",
                        tool_input_template={
                            "filename": unique_filename,
                            "file_type": "pdf",
                            "content": ""
                        },
                        depends_on=[1]
                    )
                ]
                plan = ExecutionPlan(
                    objective=objective,
                    steps=steps,
                    estimated_steps=2
                )
                return {
                    "execution_plan": plan.model_dump(),
                    "status": "PLANNED"
                }

            elif output_format in ["pdf", "markdown"]:
                unique_filename = report_builder.generate_unique_filename(task_type, doc_name, output_format)
                steps = [
                    PlanStep(
                        step_index=1,
                        description=f"Retrieve and synthesize {task_type.replace('_', ' ')} from {doc_name}",
                        agent_name="Research",
                        tool_name="rag",
                        tool_input_template={"query": prompt, "source": doc_name},
                        depends_on=[]
                    ),
                    PlanStep(
                        step_index=2,
                        description=f"Compile and generate formatted document: {unique_filename}",
                        agent_name="Execution",
                        tool_name="file_gen",
                        tool_input_template={
                            "filename": unique_filename,
                            "file_type": output_format,
                            "content": ""
                        },
                        depends_on=[1]
                    )
                ]
                plan = ExecutionPlan(
                    objective=objective,
                    steps=steps,
                    estimated_steps=2
                )
                return {
                    "execution_plan": plan.model_dump(),
                    "status": "PLANNED"
                }

            elif output_format == "chat":
                # Chat only - no file_gen, no email
                steps = [
                    PlanStep(
                        step_index=1,
                        description=f"Retrieve knowledge and answer user question from {doc_name}",
                        agent_name="Research",
                        tool_name="rag",
                        tool_input_template={"query": prompt, "source": doc_name},
                        depends_on=[]
                    )
                ]
                plan = ExecutionPlan(
                    objective=objective,
                    steps=steps,
                    estimated_steps=1
                )
                return {
                    "execution_plan": plan.model_dump(),
                    "status": "PLANNED"
                }

        conversation_context = state.get("conversation_context", "")
        system_prompt = (
            "You are the Dynamic Planner Agent in an autonomous multi-agent AI system. "
            "Your objective is to generate an executable, ordered sequence of atomic steps to accomplish the task.\n"
            "Each step must specify:\n"
            "- step_index (integer starting from 1)\n"
            "- description (concrete action to perform)\n"
            "- agent_name (Research, Execution, or Memory)\n"
            "- tool_name (calculator, search, data_analysis, file_gen, email, rag, memory, or null if direct synthesis)\n"
            "- tool_input_template (dictionary of parameters for the tool)\n"
            "- depends_on (list of prior step_indexes that must complete first)\n\n"
            "- memory: Use for retrieving prior questions, answers, or conversational context from session history (query: '...').\n"
            "- data_analysis: Use for computing dataset summaries (action: 'summary') AND generating visualization plots/charts (action: 'plot', chart_type: 'line'/'bar'/'pie'/'scatter'). Always use data_analysis (NOT rag) for tabular datasets and chart generation.\n"
            "- file_gen: Mandatory whenever generating a report, summary, PDF document, or deliverable (filename: '...', content: '...', file_type: 'pdf'). ALWAYS generate the file with file_gen before sending it via email.\n"
            "- calculator: Use for evaluating mathematical expressions (expression: '...').\n"
            "- email: Use for dispatching emails (recipient: '...', subject: '...', body: '...'). If a report, summary, or PDF was requested, email MUST depend on the prior file_gen step so the PDF file is attached and sent.\n"
            "- rag: Use for querying uploaded files or knowledge documents (query: '...', source: '...'). When the user asks a question about a document or a follow-up question on document details, ALWAYS schedule a step with agent_name: 'Research' and tool_name: 'rag'.\n"
            "- search: Use for querying public internet (query: '...')."
        )

        user_content = (
            f"Original Request: {prompt}\n"
            f"Task Specification:\n"
            f"- Intent: {task_spec.get('intent')}\n"
            f"- Task Type: {task_spec.get('task_type')}\n"
            f"- Objective: {task_spec.get('objective')}\n"
            f"- Output Format: {task_spec.get('output_format')}\n"
            f"- Send Email: {task_spec.get('send_email')}\n"
            f"- Recipient: {task_spec.get('recipient')}\n"
            f"- Tools Needed: {task_spec.get('required_tools')}\n"
            f"- Constraints: {task_spec.get('constraints')}\n"
        )
        if attached_files:
            user_content += f"Attached Files: {', '.join(attached_files)}\n"
        if conversation_context:
            safe_conv = conversation_context[-1500:] if len(conversation_context) > 1500 else conversation_context
            user_content += f"Recent Conversation History:\n{safe_conv}\n"
        if memory_context:
            safe_mem = memory_context[:1000] if len(memory_context) > 1000 else memory_context
            user_content += f"User Profile & Stored Preferences:\n{safe_mem}\n"

        if retry_instructions:
            user_content += (
                f"\nIMPORTANT: Prior execution was evaluated as flawed or incomplete.\n"
                f"Evaluator Instructions for Correction:\n{retry_instructions}\n"
                f"Please update the plan to directly resolve these issues!"
            )

        plan: ExecutionPlan = await self.llm.ainvoke_structured(
            prompt=user_content,
            schema=ExecutionPlan,
            system_prompt=system_prompt,
            fast=False
        )

        return {
            "execution_plan": plan.model_dump(),
            "status": "PLANNED"
        }

planner_agent = PlannerAgent()
