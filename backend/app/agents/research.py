import os
import re
import logging
from typing import Dict, Any, List, Optional
from backend.app.agents.base import BaseAgent
from backend.app.rag.vectorstore import rag_store
from backend.app.tools.search import SearchTool
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    """Agent 3: Conducts semantic RAG retrieval and web information synthesis."""

    def __init__(self):
        super().__init__(
            name="ResearchAgent",
            description="Retrieves domain knowledge from ChromaDB RAG store and external search."
        )

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("current_step_query") or state.get("user_prompt", "")
        user_id = state.get("user_id", "default_user")
        attached_files = list(state.get("attached_files", []))
        user_prompt = state.get("user_prompt", "")
        if not attached_files and "[Attached Files:" in user_prompt:
            matches = re.findall(r"\[Attached Files:\s*([^\]]+)\]", user_prompt)
            for m in matches:
                for p in m.split(","):
                    p_clean = p.strip()
                    if p_clean:
                        attached_files.append(p_clean)

        # 1. Auto-index any attached files if not yet in vector store
        if attached_files:
            for af in attached_files:
                if os.path.exists(af):
                    fname = os.path.basename(af)
                    try:
                        existing = rag_store.collection.get(where={"source": fname})
                        if not existing or not existing.get("ids"):
                            from backend.app.rag.ingestion import ingestion_service
                            chunks = ingestion_service.load_file(af, user_id=user_id)
                            if chunks:
                                rag_store.add_documents(chunks)
                                logger.info(f"Auto-indexed attached document '{fname}' ({len(chunks)} chunks)")
                    except Exception as ex:
                        logger.warning(f"Could not auto-index attached file {af}: {ex}")

        # 2. Contextualize query if follow-up reference in conversation
        conv_ctx = state.get("conversation_context", "")
        if conv_ctx and any(w in query.lower() for w in ["previous", "prior", "last", "second", "third", "it", "they", "that", "elaborate", "explain more", "what about"]):
            try:
                rewritten_q = await self.llm.ainvoke_text(
                    prompt=f"Conversation Context:\n{conv_ctx[-1000:]}\n\nFollow-up Request: {query}\n\nReformulate into a standalone search query for document retrieval:",
                    system_prompt="Reformulate follow-up questions into standalone search queries based on chat context. Output ONLY the query string.",
                    fast=True
                )
                if rewritten_q and len(rewritten_q.strip()) > 3:
                    query = rewritten_q.strip().strip('"').strip("'")
            except Exception as e:
                logger.warning(f"Failed to contextualize query in ResearchAgent: {e}")

        # 3. Detect section intent from query or user_prompt
        combined_text = f"{query} {state.get('user_prompt', '')}".lower()
        target_section = None
        if any(w in combined_text for w in ["project", "projects", "academic project", "personal project"]):
            target_section = "PROJECTS"
        elif any(w in combined_text for w in ["skill", "skills", "tech stack", "technologies", "programming"]):
            target_section = "SKILLS SUMMARY"
        elif any(w in combined_text for w in ["intern", "internship", "experience", "work experience", "job"]):
            target_section = "INTERNSHIP EXPERIENCE"
        elif any(w in combined_text for w in ["education", "degree", "college", "university", "academic", "b.tech", "gpa", "cgpa"]):
            target_section = "EDUCATION"
        elif any(w in combined_text for w in ["certificate", "certifications", "certification", "course"]):
            target_section = "CERTIFICATIONS"
        elif any(w in combined_text for w in ["about me", "summary", "profile", "bio", "objective", "who is"]):
            target_section = "ABOUT ME"

        # 3. Detect target document(s)
        task_spec = state.get("task_spec", {})
        task_type = task_spec.get("task_type") or "summary"
        doc_ids_from_spec = task_spec.get("document_ids", [])

        target_sources = []
        if attached_files:
            target_sources = [os.path.basename(af) for af in attached_files]
        elif doc_ids_from_spec:
            target_sources = list(doc_ids_from_spec)
        elif conv_ctx:
            # Check conversation context for recently referenced documents
            ctx_matches = re.findall(r"([a-zA-Z0-9_\-\.]+(?:\s*\(\d+\))?\.(?:pdf|docx|txt|csv|xlsx|json))", conv_ctx, flags=re.IGNORECASE)
            for m in reversed(ctx_matches):
                m_clean = m.strip()
                if os.path.exists(settings.UPLOAD_DIR):
                    for fname in os.listdir(settings.UPLOAD_DIR):
                        if fname.lower() == m_clean.lower() and fname not in target_sources:
                            target_sources.append(fname)
                            break
                if target_sources:
                    break

        if not target_sources and os.path.exists(settings.UPLOAD_DIR):
            combined_q = f"{query} {user_prompt}".lower()
            # 1. Check for explicit filenames in prompt
            file_matches = re.findall(r"([a-zA-Z0-9_\-\.]+(?:\s*\(\d+\))?\.(?:pdf|docx|txt|csv|xlsx|json))", combined_q, flags=re.IGNORECASE)
            for m in file_matches:
                m_clean = m.strip()
                for fname in os.listdir(settings.UPLOAD_DIR):
                    if fname.lower() == m_clean.lower() and fname not in target_sources:
                        target_sources.append(fname)

            # 2. Check for stem matches
            if not target_sources:
                for fname in os.listdir(settings.UPLOAD_DIR):
                    if fname.startswith("."):
                        continue
                    stem = os.path.splitext(fname)[0].lower()
                    clean_stem = re.sub(r"[^a-zA-Z0-9]+", " ", stem).strip()
                    if clean_stem in combined_q or fname.lower() in combined_q:
                        target_sources.append(fname)
                        break

            # 3. If prompt refers to document/summary/deliverables without naming a specific file:
            # Auto-associate the most recently uploaded user document
            from backend.app.services.report_builder import should_auto_associate_upload
            if not target_sources and should_auto_associate_upload(combined_q):
                recent_uploads = sorted(
                    [
                        f for f in os.listdir(settings.UPLOAD_DIR)
                        if not f.startswith(".") and os.path.isfile(os.path.join(settings.UPLOAD_DIR, f))
                    ],
                    key=lambda x: os.path.getmtime(os.path.join(settings.UPLOAD_DIR, x)),
                    reverse=True
                )
                if recent_uploads:
                    target_sources.append(recent_uploads[0])
                    logger.info(f"Target source defaulted to latest uploaded file: {recent_uploads[0]}")

        target_source = target_sources[0] if target_sources else None

        from backend.app.services.report_builder import should_auto_associate_upload
        combined_q = f"{query} {user_prompt}".lower()
        is_doc_task = bool(target_sources or attached_files or should_auto_associate_upload(combined_q))
        if not is_doc_task:
            clean_q = re.sub(
                r'(?:and\s+)?(?:send|email|mail|dispatch|forward)\s+(?:it|them|this|the\s+report|the\s+result)?\s*(?:to\s+[\w\.-]+@[\w\.-]+\.\w+.*)?$',
                '', query, flags=re.IGNORECASE
            ).strip()
            synthesis_prompt = (
                f"Topic / Request: {clean_q or query}\n\n"
                f"Instructions:\n"
                f"- Generate the requested content directly, thoroughly, and professionally.\n"
                f"- If asked for titles, produce the exact requested number of creative titles formatted as a clean numbered list.\n"
                f"- Do NOT mention email sending, external addresses, AI capabilities, or disclaimers.\n"
                f"- Start directly with the content."
            )
            findings = await self.llm.ainvoke_text(
                prompt=synthesis_prompt,
                system_prompt="You are an expert AI orchestrator and creative writer. Deliver clean, publication-ready markdown output directly addressing the topic without meta-commentary.",
                fast=False
            )
            # Remove any leading disclaimer about email capabilities if model generated one
            findings_clean = re.sub(r'^(?:I cannot send emails[^\n]*\n+|However,[^\n]*\n+)', '', findings.strip(), flags=re.IGNORECASE).strip()
            return {
                "research_findings": findings_clean or findings,
                "citations": [],
                "context_chunks_count": 0
            }

        # 4. Task-Aware Retrieval Strategy
        rag_results = []
        is_full_doc_task = task_type in ["summary", "detailed_summary", "key_points", "extraction", "table", "report", "explanation"]

        if len(target_sources) > 1 or task_type == "comparison":
            # Multi-document retrieval
            for src in target_sources:
                src_chunks = rag_store.get_document_chunks(source=src)
                if not src_chunks:
                    src_chunks = rag_store.query(query_text=query, top_k=10, source=src)
                rag_results.extend(src_chunks)
        elif target_source and is_full_doc_task:
            # Full-document sequential chunk retrieval for broad coverage
            rag_results = rag_store.get_document_chunks(source=target_source)
            if not rag_results:
                rag_results = rag_store.query(query_text=query, top_k=20, source=target_source)
        elif target_source:
            # Targeted question answering: top-k similarity search
            rag_results = rag_store.query(
                query_text=query,
                top_k=6,
                source=target_source
            )
            if not rag_results and target_section:
                rag_results = rag_store.query(
                    query_text=query,
                    top_k=6,
                    source=target_source,
                    section=target_section
                )

        # 5. Collection-wide fallback only if no target source exists
        if not rag_results and not target_source:
            if target_section:
                rag_results = rag_store.query(
                    query_text=query,
                    top_k=6,
                    user_id=user_id,
                    section=target_section
                )
            if not rag_results:
                rag_results = rag_store.query(
                    query_text=query,
                    top_k=6,
                    user_id=user_id
                )

        # 6. Fall back to SearchTool ONLY for general queries without attached documents
        web_results = []
        if not rag_results and not target_source and not attached_files:
            search_res = SearchTool.execute(query=query, max_results=3)
            if search_res.success and isinstance(search_res.result, list):
                web_results = search_res.result

        # 7. Collect context chunks and citations
        context_chunks = []
        citations = []
        seen_citations = set()
        for r in rag_results:
            context_chunks.append(r["content"])
            src_name = r.get("source") or r.get("filename") or "Document"
            page_val = r.get("page")
            page_str = f" — Page {page_val}" if page_val is not None else ""
            cite = f"{src_name}{page_str}"
            if cite not in seen_citations:
                seen_citations.add(cite)
                citations.append(cite)

        for w in web_results:
            context_chunks.append(w.get("snippet", ""))
            if w.get("url") and w.get("url") not in seen_citations:
                seen_citations.add(w.get("url"))
                citations.append(w.get("url"))

        if not context_chunks:
            if not target_source and not attached_files:
                memory_context = state.get("memory_context")
                synthesis_prompt = f"USER REQUEST:\n{query or user_prompt}\n\n"
                if memory_context:
                    synthesis_prompt += f"USER PREFERENCES & FORMATTING DIRECTIVES (STRICTLY MANDATORY):\n{memory_context}\n\n"
                synthesis_prompt += "Fulfill the request thoroughly and accurately, strictly adhering to any user preferences, summary styling, or persona directives."

                system_prompt = (
                    "You are an expert technical and creative researcher in an autonomous AI platform.\n"
                    "CRITICAL PERSONALIZATION DIRECTIVES (MANDATORY TO FOLLOW):\n"
                    "1. If user preferences specify a summary style (e.g. 'Always give 5 short bullet points with key takeaways'), "
                    "YOU MUST STRICTLY FOLLOW THAT FORMAT. Produce precisely that format (e.g. exactly 5 short bullet points with key takeaways) without long extraneous unrequested text or sections.\n"
                    "2. If user profile facts (such as college, name, organization) are recorded, reflect them accurately if relevant.\n"
                    "3. Adhere to any specified tone, style, or persona directives.\n"
                    "4. Do not include meta-commentary."
                )

                findings = await self.llm.ainvoke_text(
                    prompt=synthesis_prompt,
                    system_prompt=system_prompt,
                    fast=False
                )
                return {
                    "research_findings": findings.strip(),
                    "citations": [],
                    "context_chunks_count": 0
                }
            return {
                "research_findings": "I couldn't find enough relevant information in the uploaded documents to answer this question.",
                "citations": [],
                "context_chunks_count": 0
            }

        # 8. Construct dedicated prompt based on task type
        doc_display_name = ", ".join(target_sources) if target_sources else "the uploaded document"
        if task_type in ["summary", "detailed_summary"]:
            system_prompt = (
                "You are summarizing an uploaded document.\n"
                "Create a faithful, comprehensive summary of the provided document content.\n"
                "Requirements:\n"
                "- Strictly obey any user summary style preferences (e.g. exactly 5 short bullet points with key takeaways if specified in user directives).\n"
                "- Cover the major topics and sections across the entire document.\n"
                "- Preserve important technical details, specifications, names, numbers, and metrics.\n"
                "- Do not invent information or include unrelated generic project deliverables.\n"
                "- Do not summarize only the first chunk; synthesize information across all document sections.\n"
                "- Avoid repetitive statements.\n"
                "- Organize the result using clear markdown headings and structured bullet points."
            )
        elif task_type == "key_points":
            system_prompt = (
                "You are a document analysis assistant extracting key takeaways.\n"
                "Extract the primary key points, core findings, and critical highlights from the provided document.\n"
                "Requirements:\n"
                "- Cover key points across all major sections of the document.\n"
                "- Use concise, high-impact bullet points under organized headings.\n"
                "- Preserve critical metrics, technical terms, and dates.\n"
                "- Strictly ground every point in the provided document context."
            )
        elif task_type == "extraction":
            system_prompt = (
                "You are a precise data extraction specialist.\n"
                "Extract all requested items (such as technologies, deliverables, requirements, or metrics) from the document.\n"
                "Requirements:\n"
                "- Extract items comprehensively across the entire document context.\n"
                "- Deduplicate extracted entities.\n"
                "- For each extracted item, provide a concise explanation or context from the document.\n"
                "- Preserve technical accuracy and do not hallucinate."
            )
        elif task_type == "table":
            system_prompt = (
                "You are a structured data analyst.\n"
                "Extract the requested information from the document and format it into a clean, comprehensive Markdown table.\n"
                "Requirements:\n"
                "- Define clear, appropriate column headers.\n"
                "- Ensure every row and cell is grounded directly in the document context.\n"
                "- Preserve exact numbers, statuses, and specifications."
            )
        elif task_type == "report":
            system_prompt = (
                "You are a technical reporting specialist preparing an executive-quality report on the uploaded document.\n"
                "Structure the report with:\n"
                "# [Document Title / Executive Report]\n"
                "## 1. Executive Summary\n"
                "## 2. Technical Architecture & Main Sections\n"
                "## 3. Detailed Specifications & Requirements\n"
                "## 4. Key Findings & Metrics\n"
                "## 5. Conclusions & Next Steps\n"
                "Rules:\n"
                "- Base every section strictly on the provided document context.\n"
                "- Do not fabricate sections or insert generic project filler."
            )
        elif task_type == "explanation":
            system_prompt = (
                "You are an expert technical communicator.\n"
                "Explain the provided document in simple, clear, and plain language without losing core technical meaning.\n"
                "Requirements:\n"
                "- Use clear analogies and accessible language where helpful.\n"
                "- Break down complex concepts into digestible explanations.\n"
                "- Ground all statements strictly in the document context."
            )
        elif task_type == "comparison":
            system_prompt = (
                "You are an analytical comparative research assistant.\n"
                "Compare the provided documents objectively.\n"
                "Requirements:\n"
                "- Highlight commonalities, key differences, and unique attributes of each document.\n"
                "- Use comparison tables and structured headings.\n"
                "- Attribute findings accurately to their respective document sources."
            )
        else:
            system_prompt = (
                "You are a document analysis assistant. The user uploaded document context and requested a specific task.\n"
                "Rules:\n"
                "1. Base the result strictly on the provided document content.\n"
                "2. Do not invent information or use unrelated documents.\n"
                "3. Follow the user's requested operation exactly.\n"
                "4. If information is missing, clearly state that it was not found.\n"
                "5. Preserve important names, numbers, dates, technical terms, and metrics."
            )

        # Budget context chunks so total input tokens stay comfortably within Groq's 7,000 ITPM limit
        max_context_chars = 13500
        total_chars = sum(len(c) for c in context_chunks)
        if total_chars > max_context_chars and context_chunks:
            budget_per_chunk = max(350, max_context_chars // len(context_chunks))
            budgeted_chunks = [c[:budget_per_chunk].strip() for c in context_chunks]
        else:
            budgeted_chunks = list(context_chunks)

        user_content = (
            f"USER TASK:\n{user_prompt or query}\n\n"
            f"DOCUMENT CONTEXT ({doc_display_name}):\n" + "\n---\n".join(budgeted_chunks) + "\n\n"
            f"SOURCES:\n{', '.join(citations) if citations else 'No sources cited'}\n\n"
        )
        conv_context = state.get("conversation_context")
        if conv_context:
            user_content += f"RECENT CONVERSATION CONTEXT:\n{conv_context[-1200:]}\n\n"
        memory_context = state.get("memory_context")
        if memory_context:
            user_content += f"USER PREFERENCES & FORMATTING DIRECTIVES (STRICTLY FOLLOW):\n{memory_context}\n\n"
        user_content += "Generate the requested result faithful to the document, strictly adhering to any user preferences or summary styling."

        synthesis = await self.llm.ainvoke_text(
            prompt=user_content,
            system_prompt=system_prompt,
            fast=False
        )

        clean_synthesis = synthesis.strip()
        clean_synthesis = re.sub(r'^```(?:markdown)?\s*\n', '', clean_synthesis)
        clean_synthesis = re.sub(r'\n```$', '', clean_synthesis)

        return {
            "research_findings": clean_synthesis,
            "citations": citations,
            "context_chunks_count": len(context_chunks),
            "target_source": target_source,
            "target_sources": target_sources
        }

research_agent = ResearchAgent()
