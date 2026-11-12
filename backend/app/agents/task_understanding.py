from typing import Dict, Any
from backend.app.agents.base import BaseAgent
from backend.app.models.schemas import TaskUnderstandingResult

class TaskUnderstandingAgent(BaseAgent):
    """Agent 1: Interprets natural language requests, determines intent, tools, and constraints."""

    def __init__(self):
        super().__init__(
            name="TaskUnderstandingAgent",
            description="Analyzes user instructions to extract intent, entities, required tools, and constraints."
        )

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = state.get("user_prompt", "")
        memory_context = state.get("memory_context", "")
        attached_files = state.get("attached_files", [])

        det_result = state.get("deterministic_memory_result")
        if det_result:
            return {
                "task_spec": {
                    "intent": "Retrieve prior conversation message from session history",
                    "task_type": "memory_query",
                    "objective": f"Answer conversational memory query: {prompt}",
                    "entities": [],
                    "required_tools": ["memory"],
                    "complexity": "simple",
                    "constraints": ["Retrieve directly from conversation session messages"]
                },
                "status": "UNDERSTOOD"
            }

        conversation_context = state.get("conversation_context", "")
        system_prompt = (
            "You are the Task Understanding Agent in an autonomous multi-agent AI system. "
            "Your objective is to analyze the user's natural language request and decompose it into a formal specification.\n"
            "Identify:\n"
            "1. Intent (what the user ultimately wants)\n"
            "2. Task Type: choose precisely from:\n"
            "   - 'summary': General document summary\n"
            "   - 'detailed_summary': In-depth, comprehensive document summary\n"
            "   - 'key_points': Key bullet points, takeaways, highlights\n"
            "   - 'extraction': Specific entity, technology, or deliverable extraction\n"
            "   - 'table': Structured table or matrix generation\n"
            "   - 'report': Formal detailed report\n"
            "   - 'explanation': Plain-language or simplified explanation\n"
            "   - 'comparison': Comparative analysis between multiple documents\n"
            "   - 'question_answering': Specific question answered from document\n"
            "   - 'calculation': Math or arithmetic expression\n"
            "   - 'data_analysis': Tabular dataset analytics or plotting\n"
            "   - 'memory_query': Session conversation history retrieval (ONLY for queries asking about the chat history itself, e.g. 'what was my last question?')\n"
            "   - 'general': Web research or general assistant task\n"
            "CRITICAL FOLLOW-UP DIRECTIVE: If the user asks a question about document content (e.g. 'what is the course name?', 'what are the marks?', 'who is the student?') and recent conversation history discusses a document, classify as 'question_answering' using the 'rag' tool.\n"
            "3. Objective (clear, concise 1-sentence goal)\n"
            "4. Entities (filenames, parameters, metrics, entities)\n"
            "5. Required Tools (choose from: calculator, search, data_analysis, file_gen, email, rag, memory)\n"
            "6. Output Format: 'chat' (normal conversation response), 'pdf' (downloadable PDF file), or 'markdown'\n"
            "7. Send Email: boolean (true if user wants the result emailed)\n"
            "8. Recipient: email address if specified\n"
            "9. Detail Level: 'brief', 'medium', or 'detailed'\n"
            "10. Document IDs: list of target document filenames or IDs\n"
            "11. Constraints (formatting, specific rules, deadlines)"
        )

        user_content = f"User Request: {prompt}\n"
        if attached_files:
            user_content += f"Attached Files: {', '.join(attached_files)}\n"
        if conversation_context:
            safe_conv = conversation_context[-1500:] if len(conversation_context) > 1500 else conversation_context
            user_content += f"Recent Conversation History:\n{safe_conv}\n"
        if memory_context:
            safe_mem = memory_context[:1000] if len(memory_context) > 1000 else memory_context
            user_content += f"User Profile & Past Preferences:\n{safe_mem}\n"

        spec: TaskUnderstandingResult = await self.llm.ainvoke_structured(
            prompt=user_content,
            schema=TaskUnderstandingResult,
            system_prompt=system_prompt,
            fast=True
        )

        # Resolve active documents
        import os
        import re
        from backend.app.core.config import settings

        active_files = list(attached_files or [])
        # Check prompt for explicit filenames
        doc_in_prompt_m = re.findall(r"([a-zA-Z0-9_\-\.]+(?:\s*\(\d+\))?\.(?:pdf|docx|txt|csv|xlsx|json))", prompt, flags=re.IGNORECASE)
        for cand in doc_in_prompt_m:
            cand_clean = cand.strip()
            if cand_clean not in active_files:
                active_files.append(cand_clean)

        # Check recent conversation history for active document references if not in prompt or attachments
        if not active_files and conversation_context:
            doc_in_ctx_m = re.findall(r"([a-zA-Z0-9_\-\.]+(?:\s*\(\d+\))?\.(?:pdf|docx|txt|csv|xlsx|json))", conversation_context, flags=re.IGNORECASE)
            for cand in reversed(doc_in_ctx_m):
                cand_clean = cand.strip()
                if cand_clean not in active_files:
                    active_files.append(cand_clean)
                    break

        # Auto-discover recent upload ONLY if prompt explicitly refers to an uploaded document or deliverables
        from backend.app.services.report_builder import should_auto_associate_upload
        is_doc_task = should_auto_associate_upload(prompt, attached_files)
        if not active_files and is_doc_task and os.path.exists(settings.UPLOAD_DIR):
            recent_uploads = sorted(
                [
                    f for f in os.listdir(settings.UPLOAD_DIR)
                    if not f.startswith(".") and os.path.isfile(os.path.join(settings.UPLOAD_DIR, f))
                ],
                key=lambda f: os.path.getmtime(os.path.join(settings.UPLOAD_DIR, f)),
                reverse=True
            )
            if recent_uploads:
                active_files = [recent_uploads[0]]

        # Populate document_ids and ensure RAG tool routing for document inquiries
        clean_doc_ids = [os.path.basename(f) for f in active_files if f]
        if clean_doc_ids:
            spec.document_ids = clean_doc_ids
            if not any(t in spec.required_tools for t in ["data_analysis", "calculator"]):
                if "rag" not in spec.required_tools:
                    spec.required_tools.append("rag")
                if "memory" in spec.required_tools and spec.task_type == "memory_query":
                    spec.required_tools.remove("memory")
                    spec.task_type = "question_answering"
                    spec.objective = f"Retrieve information from {clean_doc_ids[0]} to answer: {prompt}"

        # Email & Recipient detection
        p_lower = prompt.lower()
        email_requested = any(w in p_lower for w in ["email", "mail", "send to", "send it", "forward to"]) or bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", prompt))
        if email_requested:
            spec.send_email = True
            email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", prompt)
            if email_match:
                spec.recipient = email_match.group(0)
            elif not spec.recipient:
                spec.recipient = "user@example.com"
            spec.output_format = "pdf"
            if "file_gen" not in spec.required_tools:
                spec.required_tools.append("file_gen")
            if "email" not in spec.required_tools:
                spec.required_tools.append("email")
        else:
            spec.send_email = False
            # Check if PDF output was explicitly requested without email
            pdf_explicit = any(w in p_lower for w in ["create a pdf", "generate a pdf", "save as pdf", "downloadable pdf", "pdf report", "generate report", "create report", "markdown report"])
            if pdf_explicit:
                spec.output_format = "pdf"
                if "file_gen" not in spec.required_tools:
                    spec.required_tools.append("file_gen")
            else:
                spec.output_format = "chat"
                # Strip file_gen and email for pure chat Q&A / summary
                spec.required_tools = [t for t in spec.required_tools if t not in ["file_gen", "email"]]

        # Ensure RAG tool is included if an active document exists
        has_doc = any(f.lower().endswith((".pdf", ".docx", ".txt")) for f in active_files)
        has_tabular = any(f.lower().endswith((".csv", ".xlsx", ".xls", ".json")) for f in active_files)
        if has_doc and is_doc_task:
            if "rag" not in spec.required_tools:
                spec.required_tools.insert(0, "rag")
        elif has_tabular and not has_doc:
            if "data_analysis" not in spec.required_tools:
                spec.required_tools.insert(0, "data_analysis")

        # Refine task_type if LLM generalized too much
        if any(w in p_lower for w in ["title", "titles"]):
            spec.task_type = "titles"
        elif has_doc or is_doc_task:
            if any(w in p_lower for w in ["key point", "key points", "bullet point", "takeaway", "highlights"]):
                spec.task_type = "key_points"
            elif any(w in p_lower for w in ["extract", "technologies", "tech stack", "all deliverables", "list all"]):
                spec.task_type = "extraction"
            elif any(w in p_lower for w in ["table", "matrix", "tabular"]):
                spec.task_type = "table"
            elif any(w in p_lower for w in ["detailed report", "create a report", "full report"]):
                spec.task_type = "report"
            elif any(w in p_lower for w in ["explain", "simple language", "plain english", "simplify"]):
                spec.task_type = "explanation"
            elif any(w in p_lower for w in ["compare", "comparison", "difference"]):
                spec.task_type = "comparison"
            elif any(w in p_lower for w in ["detailed summary", "comprehensive summary"]):
                spec.task_type = "detailed_summary"
            elif any(w in p_lower for w in ["summarize", "summary", "overview"]):
                spec.task_type = "summary"
            elif any(w in p_lower for w in ["what", "who", "where", "how", "when", "why", "which", "is there", "does it"]):
                spec.task_type = "question_answering"

        return {
            "task_spec": spec.model_dump(),
            "status": "UNDERSTOOD"
        }

task_understanding_agent = TaskUnderstandingAgent()
