import os
import uuid
from typing import Dict, Any, Optional
from backend.app.core.config import settings
from backend.app.models.schemas import ToolExecutionResult

class FileGenTool:
    """Tool to generate downloadable files (Markdown reports, CSV data, TXT summaries, JSON)."""

    @classmethod
    def execute(
        cls,
        content: Optional[str] = None,
        filename: Optional[str] = None,
        file_type: str = "markdown", # markdown, csv, txt, json
        **kwargs
    ) -> ToolExecutionResult:
        try:
            # Normalize content from common aliases
            actual_content = (
                content
                or kwargs.get("text")
                or kwargs.get("data")
                or kwargs.get("report_content")
                or kwargs.get("body")
                or kwargs.get("summary")
            )
            if not actual_content or not str(actual_content).strip():
                return ToolExecutionResult(
                    tool_name="file_gen",
                    success=False,
                    result=None,
                    error="File generation aborted: no substantive content provided.",
                    artifacts=[]
                )
            if isinstance(actual_content, (dict, list)):
                import json
                actual_content = json.dumps(actual_content, indent=2)

            # Normalize filename from aliases or file_path if targeting an output file
            raw_filename = (
                filename
                or kwargs.get("file_name")
                or kwargs.get("name")
                or kwargs.get("title")
            )
            if not raw_filename and kwargs.get("file_path"):
                cand = os.path.basename(str(kwargs.get("file_path")))
                if cand.endswith((".md", ".txt", ".csv", ".json")):
                    raw_filename = cand

            extensions = {
                "markdown": "md",
                "pdf": "pdf",
                "csv": "csv",
                "txt": "txt",
                "json": "json"
            }
            requested_type = file_type.lower()
            ext = extensions.get(requested_type, "md")
            if not raw_filename:
                raw_filename = f"report_{uuid.uuid4().hex[:8]}.{ext}"
            elif not raw_filename.endswith(f".{ext}") and "." not in raw_filename:
                raw_filename = f"{raw_filename}.{ext}"

            clean_filename = os.path.basename(raw_filename)
            base_stem, current_ext = os.path.splitext(clean_filename)
            file_path = os.path.join(settings.OUTPUT_DIR, clean_filename)

            artifacts = []
            pdf_path = None

            if current_ext.lower() in [".md", ".txt", ".pdf"] or requested_type in ["markdown", "pdf", "txt"]:
                from backend.app.tools.pdf_compiler import compile_markdown_to_pdf

                # 1. Save markdown version
                md_filename = f"{base_stem}.md"
                md_path = os.path.join(settings.OUTPUT_DIR, md_filename)
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(str(actual_content))

                # 2. Compile companion PDF document
                pdf_filename = f"{base_stem}.pdf"
                pdf_path = os.path.join(settings.OUTPUT_DIR, pdf_filename)
                try:
                    compile_markdown_to_pdf(str(actual_content), pdf_path, doc_title=base_stem.replace('_', ' ').title())
                except Exception as pdf_err:
                    import logging
                    logging.getLogger(__name__).warning(f"Could not compile companion PDF: {pdf_err}")
                    pdf_path = None

                # Primary target and artifacts ordering
                if current_ext.lower() == ".pdf" or requested_type == "pdf":
                    file_path = pdf_path if pdf_path else md_path
                    artifacts = [f for f in [pdf_path, md_path] if f and os.path.exists(f)]
                else:
                    file_path = md_path
                    artifacts = [f for f in [pdf_path, md_path] if f and os.path.exists(f)]

            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(str(actual_content))
                artifacts = [file_path]

            return ToolExecutionResult(
                tool_name="file_gen",
                success=True,
                result={
                    "filename": os.path.basename(file_path),
                    "file_path": file_path,
                    "pdf_path": pdf_path,
                    "size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else len(str(actual_content)),
                    "file_type": file_type
                },
                error=None,
                artifacts=artifacts
            )
        except Exception as e:
            return ToolExecutionResult(
                tool_name="file_gen",
                success=False,
                result=None,
                error=str(e),
                artifacts=[]
            )
