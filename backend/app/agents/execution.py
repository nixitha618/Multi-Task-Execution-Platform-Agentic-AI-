import os
from typing import Dict, Any, List, Optional
import logging
from backend.app.agents.base import BaseAgent
from backend.app.tools.registry import tool_registry
from backend.app.models.schemas import ToolExecutionResult
from backend.app.services.report_builder import report_builder

logger = logging.getLogger(__name__)

def _format_data_analysis_output(out: Dict[str, Any]) -> str:
    lines = []
    src = os.path.basename(str(out.get("dataset_source", "dataset.csv")))
    shape = out.get("shape", [])
    if len(shape) == 2:
        lines.append(f"**Dataset Source:** `{src}` ({shape[0]} rows, {shape[1]} columns)\n")
    cols = out.get("columns", [])
    if cols:
        lines.append(f"**Columns:** {', '.join([f'`{c}`' for c in cols])}\n")

    stats = out.get("statistics", {})
    if isinstance(stats, dict) and stats:
        lines.append("#### Key Statistical Metrics\n")
        lines.append("| Numeric Column | Count | Mean | Std Dev | Min | Max |")
        lines.append("|---|---|---|---|---|---|")
        for col, m in stats.items():
            if isinstance(m, dict):
                mean_v = f"{m.get('mean', 0):,.2f}" if isinstance(m.get('mean'), (int, float)) else str(m.get('mean', '-'))
                std_v = f"{m.get('std', 0):,.2f}" if isinstance(m.get('std'), (int, float)) else str(m.get('std', '-'))
                min_v = f"{m.get('min', 0):,.2f}" if isinstance(m.get('min'), (int, float)) else str(m.get('min', '-'))
                max_v = f"{m.get('max', 0):,.2f}" if isinstance(m.get('max'), (int, float)) else str(m.get('max', '-'))
                cnt_v = str(int(m.get('count', 0))) if isinstance(m.get('count'), (int, float)) else str(m.get('count', '-'))
                lines.append(f"| {col.replace('_', ' ').title()} | {cnt_v} | {mean_v} | {std_v} | {min_v} | {max_v} |")
        lines.append("")

    head = out.get("head", [])
    if isinstance(head, list) and head and isinstance(head[0], dict):
        lines.append("#### Sample Records Preview\n")
        headers = list(head[0].keys())[:7]
        lines.append("| " + " | ".join([h.replace('_', ' ').title() for h in headers]) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for item in head[:5]:
            lines.append("| " + " | ".join([str(item.get(h, '')) for h in headers]) + " |")
        lines.append("")

    return "\n".join(lines) if lines else str(out)

import re

def _interpolate_template_text(text: str, prior_results: Dict[str, Any], context_summary: str = "") -> str:
    if not text:
        return ""
    step_map = {}
    last_text_output = ""
    has_file_gen = False
    file_gen_name = ""

    if isinstance(prior_results, dict):
        for k, v in prior_results.items():
            if isinstance(v, dict):
                out = v.get("output")
                tool = v.get("tool_name")
                if tool == "file_gen" and isinstance(out, dict):
                    has_file_gen = True
                    file_gen_name = out.get("filename", "deliverables report")
                    out_str = f"The generated deliverable report ({file_gen_name}) is attached."
                elif tool == "data_analysis" and isinstance(out, dict):
                    out_str = _format_data_analysis_output(out)
                    last_text_output = out_str
                elif out is not None:
                    out_str = str(out)
                    last_text_output = out_str
                else:
                    out_str = ""

                step_map[str(k)] = out_str
                step_map[f"step_{k}"] = out_str
                step_map[f"step_{k}_output"] = out_str
                if tool:
                    step_map[f"{tool}_output"] = out_str
                    step_map[tool] = out_str

    def _repl(match):
        var_name = match.group(1).strip()
        if var_name in step_map:
            return step_map[var_name]
        if var_name in ["summary", "rag_output", "analysis", "result", "output"]:
            if last_text_output:
                return last_text_output
            if has_file_gen:
                return f"The deliverables report ({file_gen_name}) has been compiled into the attached PDF document."
            return context_summary or ""
        return ""

    res = re.sub(r'\{\{\s*([a-zA-Z0-9_\-]+)\s*\}\}', _repl, str(text))
    res = re.sub(r'\{([a-zA-Z0-9_]+_output)\}', _repl, res)
    res = re.sub(r'\[(?:Your Name|User|Sender Name|Agent Name)\]', 'Autonomous Multi-Agent AI System', res, flags=re.IGNORECASE)
    res = re.sub(r'\[(?:Project Name|System Name)\]', 'Agentic AI System', res, flags=re.IGNORECASE)
    res = re.sub(r'\[(?:Your Contact Information|Contact Information)\]', 'Autonomous AI Operations', res, flags=re.IGNORECASE)
    return res

class ExecutionAgent(BaseAgent):
    """Agent 4: Executes scheduled tool actions (Calculator, Data Analysis, Files, Email)."""

    def __init__(self):
        super().__init__(
            name="ExecutionAgent",
            description="Coordinates tool calls and executes algorithmic, file, and data processing operations."
        )

    async def execute_step(
        self,
        step: Dict[str, Any],
        prior_results: Dict[str, Any],
        user_prompt: Optional[str] = None,
        user_id: Optional[str] = None,
        memory_context: Optional[str] = None
    ) -> Dict[str, Any]:
        tool_name = step.get("tool_name")
        input_template = step.get("tool_input_template", {}) or {}
        description = step.get("description", "")

        # Enrich template with prior results or parameters if needed
        merged_inputs = dict(input_template)

        # Collect prior outputs and file artifacts across all completed steps
        prior_artifacts: List[str] = []
        prior_summaries: List[str] = []
        if isinstance(prior_results, dict):
            for k, v in sorted(prior_results.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0):
                if isinstance(v, dict):
                    for art in v.get("artifacts", []):
                        if art and str(art) not in prior_artifacts:
                            prior_artifacts.append(str(art))
                    out = v.get("output")
                    if out:
                        if v.get("tool_name") == "data_analysis" and isinstance(out, dict):
                            formatted_out = _format_data_analysis_output(out)
                        else:
                            formatted_out = str(out)
                        prior_summaries.append(f"### Output from Step {k} ({v.get('tool_name', 'prior_step')}):\n{formatted_out}")

        # Intelligent parameter interpolation and scoping per tool type
        if tool_name == "file_gen":
            user_text_lower = f"{user_prompt or ''} {description or ''}".lower()
            merged_inputs["file_type"] = "pdf"

            # 1. Discover target document, task type, and citations from prior results
            cand_doc = None
            doc_citations = []
            prior_research_out = ""

            if isinstance(prior_results, dict):
                for k, v in sorted(prior_results.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0):
                    if isinstance(v, dict):
                        if v.get("tool_name") == "rag" or "citations" in v:
                            if v.get("citations"):
                                doc_citations.extend(v["citations"])
                            if v.get("output"):
                                prior_research_out = str(v["output"])
                        if v.get("target_source"):
                            cand_doc = v["target_source"]

            if not cand_doc and user_prompt:
                doc_m = re.search(r"([\w\- \(\)]+\.(?:pdf|docx|txt|csv|xlsx|json))", user_prompt, flags=re.IGNORECASE)
                if doc_m:
                    cand_doc = doc_m.group(1).strip()
            if not cand_doc:
                from backend.app.core.config import settings
                from backend.app.services.report_builder import should_auto_associate_upload
                if should_auto_associate_upload(user_prompt) and os.path.exists(settings.UPLOAD_DIR):
                    recents = [f for f in os.listdir(settings.UPLOAD_DIR) if not f.startswith(".") and os.path.isfile(os.path.join(settings.UPLOAD_DIR, f))]
                    if recents:
                        recents.sort(key=lambda x: os.path.getmtime(os.path.join(settings.UPLOAD_DIR, x)), reverse=True)
                        cand_doc = recents[0]

            # If cand_doc matches the target output file being generated, it is NOT an input source doc
            target_out_fn = merged_inputs.get("filename")
            if cand_doc and target_out_fn and os.path.basename(cand_doc).lower() == os.path.basename(target_out_fn).lower():
                cand_doc = None

            doc_name = cand_doc

            # Determine task type
            task_type = "summary"
            if any(w in user_text_lower for w in ["key point", "key points", "bullet point", "takeaway", "highlights"]):
                task_type = "key_points"
            elif any(w in user_text_lower for w in ["extract", "technologies", "tech stack", "all deliverables"]):
                task_type = "extraction"
            elif any(w in user_text_lower for w in ["table", "matrix", "tabular"]):
                task_type = "table"
            elif any(w in user_text_lower for w in ["detailed report", "create a report", "full report"]):
                task_type = "report"
            elif any(w in user_text_lower for w in ["explain", "simple language", "plain english", "simplify"]):
                task_type = "explanation"
            elif any(w in user_text_lower for w in ["compare", "comparison", "difference"]):
                task_type = "comparison"
            elif any(w in user_text_lower for w in ["detailed summary", "comprehensive summary"]):
                task_type = "detailed_summary"

            raw_content = merged_inputs.get("content")
            if raw_content:
                raw_content = _interpolate_template_text(str(raw_content), prior_results, context_summary="\n\n".join(prior_summaries))

            has_placeholder_bracket = bool(re.search(r'\[(insert|replace|add|fill|todo|placeholder|detailed|summary).*?\]', str(raw_content), flags=re.IGNORECASE))
            is_placeholder = (
                not raw_content
                or str(raw_content).strip() in ["", "None", "..."]
                or "{{" in str(raw_content)
                or has_placeholder_bracket
                or len(str(raw_content).strip()) < 100
            )

            substantive_body = ""
            if not is_placeholder and raw_content:
                substantive_body = str(raw_content).strip()
            elif prior_research_out and len(prior_research_out.strip()) > 80:
                substantive_body = prior_research_out.strip()
            elif prior_summaries:
                substantive_body = "\n\n".join(prior_summaries).strip()

            if not substantive_body or len(substantive_body) < 80:
                # Dynamic task synthesis (never hardcoded deliverables report)
                try:
                    mem_part = f"\n\nUSER PREFERENCES & PROFILE:\n{memory_context}\n" if memory_context else ""
                    synthesis_prompt = (
                        f"You are a professional technical reporting agent.\n"
                        f"USER TASK: {user_prompt or description}\n\n"
                        f"CONTEXT:\n" + ("\n\n".join(prior_summaries) if prior_summaries else f"Context from {doc_name}") +
                        mem_part + "\n\n"
                        f"Generate the requested {task_type} grounded strictly in the context above, honoring any user formatting style or profile preferences. Do not hallucinate."
                    )
                    gen_report = await self.llm.ainvoke_text(
                        prompt=synthesis_prompt,
                        system_prompt="You produce clean, publication-ready technical Markdown without meta-commentary.",
                        fast=False
                    )
                    substantive_body = gen_report.strip()
                    substantive_body = re.sub(r'^```(?:markdown)?\s*\n', '', substantive_body)
                    substantive_body = re.sub(r'\n```$', '', substantive_body)
                except Exception as gen_err:
                    self.logger.warning(f"Task synthesis fallback error: {gen_err}")
                    substantive_body = prior_research_out or ""

            # Personalization: Ensure substantive_body strictly follows active summary_style preferences
            from backend.app.services.memory_service import memory_service
            uid = user_id or "default_user"
            user_summary_style = await memory_service.get_user_preference(uid, "summary_style")

            is_summary_task = any(w in task_type.lower() or w in user_text_lower for w in ["summary", "summarize", "key_point", "bullet", "takeaway", "overview"])
            if user_summary_style and is_summary_task and substantive_body:
                bullet_lines = [l.strip() for l in substantive_body.splitlines() if l.strip().startswith(("-", "*", "•")) or re.match(r"^\d+[\.\)]\s+", l.strip())]
                # If substantive_body is not yet formatted as the requested bullet points:
                if len(bullet_lines) < 3 or len(substantive_body) > 1000 or ("## " in substantive_body and len(bullet_lines) == 0):
                    try:
                        style_prompt = (
                            f"Format the following content strictly according to the user's styling preference.\n\n"
                            f"USER STYLING PREFERENCE (MANDATORY TO FOLLOW):\n{user_summary_style}\n\n"
                            f"SOURCE CONTENT:\n{substantive_body[:3500]}\n\n"
                            f"Generate ONLY the formatted summary (e.g. 5 short bullet points with key takeaways) without extraneous commentary or unrequested headings."
                        )
                        styled_res = await self.llm.ainvoke_text(
                            prompt=style_prompt,
                            system_prompt="You are a precise technical formatter. Format content strictly according to the user's specified summary style and constraints.",
                            fast=False
                        )
                        if styled_res and len(styled_res.strip()) > 30:
                            clean_styled = styled_res.strip()
                            clean_styled = re.sub(r'^```(?:markdown)?\s*\n', '', clean_styled)
                            clean_styled = re.sub(r'\n```$', '', clean_styled)
                            substantive_body = clean_styled
                    except Exception as style_err:
                        self.logger.warning(f"Could not apply summary style to file_gen: {style_err}")

            # Format via ReportBuilder
            report_title = report_builder.generate_report_title(
                task_type=task_type,
                doc_name=doc_name,
                prompt=user_prompt,
                content=substantive_body
            )
            formatted_report = report_builder.build_report(
                content=substantive_body,
                title=report_title,
                source_documents=[doc_name] if doc_name else None,
                task_type=task_type,
                prompt=user_prompt,
                citations=doc_citations
            )

            if not merged_inputs.get("filename") or str(merged_inputs.get("filename")).strip() in ["", "document.pdf"]:
                merged_inputs["filename"] = report_builder.generate_unique_filename(
                    task_type=task_type,
                    doc_name=doc_name,
                    prompt=user_prompt,
                    ext="pdf"
                )
            merged_inputs["content"] = formatted_report
            merged_inputs.pop("file_path", None)

        elif tool_name == "email":
            # 1. Content Grounding Validation (Section 15)
            prior_content_text = ""
            if isinstance(prior_results, dict):
                for k, v in prior_results.items():
                    if isinstance(v, dict) and v.get("output"):
                        prior_content_text += " " + str(v.get("output"))

            is_valid_grounding = report_builder.validate_content_grounding(prior_content_text)
            if not is_valid_grounding and not prior_artifacts:
                self.logger.error("Content grounding validation failed: generated content is empty or ungrounded. Cancelling email transmission.")
                return {
                    "step_index": step.get("step_index"),
                    "tool_name": "email",
                    "success": False,
                    "output": "Report generation failed because the generated content could not be grounded in the uploaded document.",
                    "error": "Content validation failed: generated content is not grounded in the uploaded document. Email was not sent.",
                    "artifacts": []
                }

            # 2. Attachments: Gather artifacts from current execution only (Section 13 & 14)
            valid_attachments = []
            for art in prior_artifacts:
                if art and os.path.isfile(art) and os.path.getsize(art) > 0:
                    if art not in valid_attachments:
                        valid_attachments.append(art)

            # Prioritize newly generated PDFs
            pdf_first = [a for a in valid_attachments if a.endswith(".pdf")] + [a for a in valid_attachments if not a.endswith(".pdf")]
            merged_inputs["attachments"] = pdf_first

            # 3. Dynamic Subject and Body (Section 20)
            user_text_lower = f"{user_prompt or ''} {description or ''}".lower()
            cand_doc = None
            doc_m = re.search(r"([\w\- \(\)]+\.(?:pdf|docx|txt|csv|xlsx|json))", user_text_lower)
            if doc_m:
                cand_doc = doc_m.group(1).strip()
            if not cand_doc:
                from backend.app.core.config import settings
                from backend.app.services.report_builder import should_auto_associate_upload
                if should_auto_associate_upload(user_prompt) and os.path.exists(settings.UPLOAD_DIR):
                    recents = [f for f in os.listdir(settings.UPLOAD_DIR) if not f.startswith(".") and os.path.isfile(os.path.join(settings.UPLOAD_DIR, f))]
                    if recents:
                        recents.sort(key=lambda x: os.path.getmtime(os.path.join(settings.UPLOAD_DIR, x)), reverse=True)
                        cand_doc = recents[0]

            topic_title = report_builder.generate_report_title("summary", doc_name=cand_doc, prompt=user_prompt)

            # Dynamic Subject
            if not merged_inputs.get("subject") or "project deliverable" in str(merged_inputs.get("subject")).lower():
                merged_inputs["subject"] = topic_title

            # Personalization: Signature handling
            uid = user_id or "default_user"
            merged_inputs["user_id"] = uid
            from backend.app.services.memory_service import memory_service
            user_sig = await memory_service.get_user_preference(uid, "signature")
            if user_sig:
                merged_inputs["signature"] = user_sig

            closing_signature = user_sig if user_sig else "Best regards,\nAutonomous Agentic AI Orchestrator"

            # Dynamic Body
            body = merged_inputs.get("body")
            if body:
                body = _interpolate_template_text(str(body), prior_results, context_summary="\n\n".join(prior_summaries))
                body = re.sub(r'\{\{\s*[a-zA-Z0-9_\-]+\s*\}\}', '', body).strip()

            if not body or str(body).strip() in ["", "None", "..."] or len(str(body).strip()) < 30 or "project deliverable" in str(body).lower():
                merged_inputs["body"] = (
                    f"Hello,\n\n"
                    f"Please find attached the generated document for: '{topic_title}'.\n\n"
                    f"{closing_signature}"
                )
            else:
                if user_sig and user_sig.lower() not in body.lower():
                    generic_closings = [
                        r"Best regards,[\s\r\n]*(?:Autonomous Agentic AI Orchestrator|Autonomous Multi-Agent AI System|Autonomous AI Operations|Agentic AI System|Autonomous AI System)",
                        r"Regards,[\s\r\n]*(?:Autonomous Agentic AI Orchestrator|Autonomous Multi-Agent AI System|Autonomous AI Operations|Agentic AI System|Autonomous AI System)",
                        r"Sincerely,[\s\r\n]*(?:Autonomous Agentic AI Orchestrator|Autonomous Multi-Agent AI System|Autonomous AI Operations|Agentic AI System|Autonomous AI System)",
                        r"Warm regards,[\s\r\n]*(?:Autonomous Agentic AI Orchestrator|Autonomous Multi-Agent AI System|Autonomous AI Operations|Agentic AI System|Autonomous AI System)",
                        r"Best,[\s\r\n]*(?:Autonomous Agentic AI Orchestrator|Autonomous Multi-Agent AI System|Autonomous AI Operations|Agentic AI System|Autonomous AI System)"
                    ]
                    replaced = False
                    for pat in generic_closings:
                        if re.search(pat, body, flags=re.IGNORECASE):
                            body = re.sub(pat, user_sig, body, flags=re.IGNORECASE).strip()
                            replaced = True
                            break
                    if not replaced:
                        body = f"{body.strip()}\n\n{user_sig}"
                merged_inputs["body"] = body

            merged_inputs.pop("file_path", None)

        elif tool_name == "calculator":
            merged_inputs.pop("file_path", None)

        elif tool_name == "data_analysis":
            if "file_path" in prior_results and "file_path" not in merged_inputs:
                merged_inputs["file_path"] = prior_results["file_path"]

            uid = user_id or "default_user"
            merged_inputs["user_id"] = uid
            if not merged_inputs.get("chart_theme"):
                from backend.app.services.memory_service import memory_service
                user_theme = await memory_service.get_user_preference(uid, "chart_theme")
                if user_theme:
                    merged_inputs["chart_theme"] = user_theme

        self.logger.info(f"Executing step {step.get('step_index')}: {description} with tool '{tool_name}'")

        if not tool_name:
            # Step is a pure reasoning/synthesis step
            synthesis = await self.llm.ainvoke_text(
                prompt=f"Execute this step: {description}\nContext from prior results: {prior_results}",
                system_prompt="You are an autonomous execution agent completing a specific operational step.",
                fast=True
            )
            return {
                "step_index": step.get("step_index"),
                "tool_name": None,
                "success": True,
                "output": synthesis,
                "artifacts": []
            }

        result: ToolExecutionResult = await tool_registry.aexecute(tool_name, **merged_inputs)

        return {
            "step_index": step.get("step_index"),
            "tool_name": tool_name,
            "success": result.success,
            "output": result.result,
            "error": result.error,
            "artifacts": result.artifacts
        }

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the next pending step in the execution plan."""
        plan = state.get("execution_plan", {})
        steps = plan.get("steps", [])
        step_results = state.get("step_results", {})
        artifacts = state.get("artifacts", [])

        # Find next unexecuted step
        completed_indices = {int(k) for k in step_results.keys()}
        next_step = None

        for s in steps:
            s_idx = s.get("step_index")
            if s_idx not in completed_indices:
                # Check dependencies
                deps = s.get("depends_on", [])
                if all(d in completed_indices for d in deps):
                    next_step = s
                    break

        if not next_step:
            return {"status": "ALL_STEPS_EXECUTED"}

        step_output = await self.execute_step(
            next_step,
            step_results,
            user_prompt=state.get("user_prompt"),
            user_id=state.get("user_id"),
            memory_context=state.get("memory_context")
        )
        s_idx = next_step.get("step_index")
        step_results[str(s_idx)] = step_output

        if step_output.get("artifacts"):
            artifacts.extend(step_output["artifacts"])

        return {
            "step_results": step_results,
            "artifacts": list(set(artifacts)),
            "last_executed_step": s_idx
        }

execution_agent = ExecutionAgent()
