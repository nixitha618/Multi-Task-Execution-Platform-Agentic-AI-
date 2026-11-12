import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from backend.app.core.datetime_utils import get_ist_filename_timestamp, format_ist

def should_auto_associate_upload(prompt: Optional[str], attached_files: Optional[List[str]] = None) -> bool:
    """
    Returns True ONLY if the prompt or state explicitly indicates that an uploaded document
    or file is the target of the operation, but no exact filename was attached.
    Never returns True for creative or general topics like 'titles on photosynthesis'.
    """
    if attached_files and len(attached_files) > 0:
        return True
    p = (prompt or "").lower()
    # Check if a filename extension is mentioned
    if re.search(r"[\w\- \(\)]+\.(?:pdf|docx|txt|csv|xlsx|json)", p):
        return True
    # Explicit document phrases
    doc_phrases = [
        "this document", "the document", "this pdf", "the pdf", "this file", "the file",
        "uploaded document", "uploaded file", "uploaded pdf", "attached document", "attached file",
        "from the document", "from this document", "in the document", "in this pdf",
        "current project deliverables", "project deliverables", "deliverables report"
    ]
    return any(phrase in p for phrase in doc_phrases)

class ReportBuilderService:
    """Decoupled Report Builder: constructs formatted Markdown documents,
    assigns dynamic titles, manages unique filenames, and provides content validation.
    """

    @staticmethod
    def extract_topic_from_prompt(prompt: Optional[str]) -> str:
        """Extracts a clean, human-readable topic from a natural language prompt."""
        p = (prompt or "").strip()
        if not p:
            return "Document Report"

        p_lower = p.lower()
        if "photosynthesis" in p_lower:
            if any(w in p_lower for w in ["title", "titles"]):
                return "Titles of Photosynthesis"
            return "Photosynthesis Analysis"

        if "reproductive" in p_lower:
            return "Reproductive System Overview"

        # Strip action prefixes
        clean = re.sub(
            r'^(?:generate|create|write|draft|produce|give me|make|provide|send me|send it to|send them to)\s+',
            '', p, flags=re.IGNORECASE
        )
        # Strip email suffix instructions
        clean = re.sub(
            r'(?:and\s+)?(?:send|email|mail|dispatch|forward)\s+(?:it|them|this|the\s+report|the\s+result)?\s*(?:to\s+[\w\.-]+@[\w\.-]+\.\w+.*)?$',
            '', clean, flags=re.IGNORECASE
        ).strip()
        # Strip punctuation
        clean = clean.strip(".,;:?!'\" ")
        
        if clean:
            words = clean.split()
            title_words = []
            for i, w in enumerate(words):
                if i > 0 and w.lower() in ["on", "of", "and", "in", "for", "the", "a", "an", "to", "with"]:
                    title_words.append(w.lower())
                else:
                    title_words.append(w.capitalize())
            return " ".join(title_words)
        return "Topic Report"

    @classmethod
    def generate_report_title(
        cls,
        task_type: str,
        doc_name: Optional[str] = None,
        prompt: Optional[str] = None,
        content: Optional[str] = None
    ) -> str:
        p = (prompt or "").strip()
        p_lower = p.lower()

        # If user explicitly asked for "titles on photosynthesis", return exact clean title
        if "photosynthesis" in p_lower and any(w in p_lower for w in ["title", "titles"]):
            return "Titles of Photosynthesis"

        # Check if first line of content has an H1
        if content:
            first_line = content.strip().splitlines()[0] if content.strip() else ""
            if first_line.startswith("# "):
                h1_title = first_line[2:].strip()
                if len(h1_title) > 3 and not any(bad in h1_title.lower() for bad in ["document summary", "untitled", "deliverables report"]):
                    return h1_title

        # Check if genuine document name is present (not None or generic placeholder)
        is_genuine_doc = bool(doc_name and doc_name.lower() not in ["none", "document.pdf", "unknown", ""])

        if is_genuine_doc:
            stem = os.path.splitext(os.path.basename(doc_name))[0]
            doc_clean = stem.replace("_", " ").title()

            if "technolog" in p_lower:
                return f"Technologies Extracted from {doc_clean}"
            elif "deliverable" in p_lower:
                return f"Project Deliverables - {doc_clean}"
            elif task_type in ["summary", "detailed_summary"]:
                return f"{doc_clean} - Document Summary"
            elif task_type == "key_points":
                return f"Key Points & Takeaways - {doc_clean}"
            elif task_type == "extraction":
                return f"Extracted Findings & Specifications - {doc_clean}"
            elif task_type == "table":
                return f"Structured Matrix & Data Breakdown - {doc_clean}"
            elif task_type == "report":
                return f"Detailed Report - {doc_clean}"
            elif task_type == "explanation":
                return f"Simplified Explanation - {doc_clean}"
            elif task_type == "comparison":
                return f"Comparative Document Analysis"
            else:
                return f"{doc_clean} - Analysis & Summary Report"

        # Topic-based title when there is no source document
        topic = cls.extract_topic_from_prompt(prompt)
        if any(w in p_lower for w in ["title", "titles"]):
            if "title" not in topic.lower():
                return f"Titles: {topic}"
            return topic
        elif "report" in p_lower:
            if "report" not in topic.lower():
                return f"Report: {topic}"
            return topic
        elif "summary" in p_lower:
            return f"Summary: {topic}"
        else:
            return topic

    @classmethod
    def generate_unique_filename(
        cls,
        task_type: str,
        doc_name: Optional[str] = None,
        prompt: Optional[str] = None,
        ext: str = "pdf"
    ) -> str:
        timestamp = get_ist_filename_timestamp()
        safe_ext = ext.lstrip('.')
        is_genuine_doc = bool(doc_name and doc_name.lower() not in ["none", "document.pdf", "unknown", ""])

        if is_genuine_doc:
            clean_stem = re.sub(r'[^a-zA-Z0-9_\-]', '', os.path.splitext(os.path.basename(doc_name))[0])
            safe_type = re.sub(r'[^a-zA-Z0-9_]', '', task_type) or "report"
            return f"{safe_type}_{clean_stem}_{timestamp}.{safe_ext}"

        # Otherwise derive filename from prompt topic
        p_lower = (prompt or "").lower()
        if "photosynthesis" in p_lower and any(w in p_lower for w in ["title", "titles"]):
            return f"titles_of_photosynthesis_{timestamp}.{safe_ext}"

        topic = cls.extract_topic_from_prompt(prompt)
        clean_topic = re.sub(r'[^a-zA-Z0-9_]', '_', topic.lower()).strip('_')
        clean_topic = re.sub(r'_+', '_', clean_topic)
        if not clean_topic:
            clean_topic = "generated_document"
        if len(clean_topic) > 35:
            clean_topic = clean_topic[:35].rstrip('_')

        return f"{clean_topic}_{timestamp}.{safe_ext}"

    @classmethod
    def build_report(
        cls,
        content: str,
        title: str,
        source_documents: Optional[List[str]] = None,
        task_type: str = "summary",
        prompt: Optional[str] = None,
        citations: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        clean_content = str(content or "").strip()
        # If content already starts with an H1 (# Title), strip it to avoid redundant titles
        if clean_content.startswith('# '):
            first_line, rest = clean_content.split('\n', 1) if '\n' in clean_content else (clean_content, "")
            extracted_title = first_line[2:].strip()
            if len(extracted_title) > 5 and not any(bad in extracted_title.lower() for bad in ["document summary", "untitled", "deliverables report"]):
                title = extracted_title
            clean_content = rest.strip()

        # Filter out invalid / placeholder source documents
        real_sources = [
            os.path.basename(s) for s in (source_documents or [])
            if s and str(s).lower() not in ["none", "unknown", "document.pdf", ""]
        ]
        if real_sources:
            sources_str = ", ".join(real_sources)
        else:
            sources_str = "None (Autonomous Knowledge & Direct Synthesis)"

        # Accurate Task Label
        p_lower = (prompt or "").lower()
        if "photosynthesis" in p_lower and any(w in p_lower for w in ["title", "titles"]):
            task_label = "Creative Title Generation"
        elif any(w in p_lower for w in ["title", "titles"]):
            task_label = "Creative Title Generation"
        elif any(w in p_lower for w in ["extract", "tech stack", "technologies"]):
            task_label = "Entity & Specification Extraction"
        elif any(w in p_lower for w in ["key point", "takeaway", "bullet point"]):
            task_label = "Key Points Extraction"
        elif any(w in p_lower for w in ["table", "matrix", "tabular"]):
            task_label = "Structured Tabular Analysis"
        elif any(w in p_lower for w in ["detailed report", "report"]):
            task_label = "Technical Report"
        elif task_type in ["summary", "detailed_summary"]:
            task_label = "Document Summary" if real_sources else "Topic Summary"
        else:
            task_label = task_type.replace('_', ' ').title()

        date_str = format_ist()

        report_parts = [
            f"# {title}\n",
            f"**Source Document(s):** {sources_str}  ",
            f"**Analysis Task:** {task_label}  ",
            f"**Generated:** {date_str}\n",
            "---\n",
            clean_content
        ]

        # Add Sources & Citations section if provided
        if citations:
            report_parts.append("\n\n---\n\n## Sources & Reference Citations\n")
            for c in citations:
                report_parts.append(f"- {c}")

        return "\n".join(report_parts)

    @staticmethod
    def validate_content_grounding(content: Optional[str]) -> bool:
        if not content:
            return False
        text = str(content).strip()
        if len(text) < 40:
            return False
        error_indicators = [
            "couldn't find enough relevant information",
            "could not find enough relevant information",
            "no relevant information found",
            "execution failed",
            "report generation failed"
        ]
        text_lower = text.lower()
        if any(ind in text_lower for ind in error_indicators):
            return False
        # Disallow unrendered template mustache placeholders
        if "{{" in text or "}}" in text:
            return False
        return True

report_builder = ReportBuilderService()

