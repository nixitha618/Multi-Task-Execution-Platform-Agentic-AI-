import os
import re
import uuid
import hashlib
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

class DocumentIngestionService:
    """Production-ready document ingestion service supporting PDF, DOCX, TXT, CSV, and XLSX.

    Features:
    - Multi-format text extraction (PDF page-by-page, DOCX paragraphs, TXT encodings, CSV/XLSX tabular)
    - Precise page-number preservation for PDFs
    - Text cleaning, unicode normalization, and paragraph preservation
    - SHA-256 file hashing for duplicate detection
    - UUID4 document_id and {document_id}_{chunk_index} chunk_id generation
    - Semantic section tagging for resumes and structured reports
    - Robust empty document and format error handling
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        self.chunk_size = chunk_size or getattr(settings, "RAG_CHUNK_SIZE", 800)
        self.chunk_overlap = chunk_overlap or getattr(settings, "RAG_CHUNK_OVERLAP", 150)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """Compute SHA-256 hash of file for duplicate prevention."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize whitespace, sanitize null bytes, and preserve paragraph structure."""
        if not text:
            return ""
        text = text.replace("\x00", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_sections(self, text: str) -> List[Dict[str, str]]:
        """Identifies semantic sections in resumes, reports, and structured documents."""
        section_patterns = [
            r"(?:^|\n)\s*(PROJECTS|ACADEMIC PROJECTS|PERSONAL PROJECTS|KEY PROJECTS)\b",
            r"(?:^|\n)\s*(SKILLS SUMMARY|TECHNICAL SKILLS|SKILLS|CORE COMPETENCIES)\b",
            r"(?:^|\n)\s*(INTERNSHIP EXPERIENCE|WORK EXPERIENCE|EXPERIENCE|PROFESSIONAL EXPERIENCE)\b",
            r"(?:^|\n)\s*(EDUCATION|ACADEMIC BACKGROUND)\b",
            r"(?:^|\n)\s*(CERTIFICATIONS|COURSES & CERTIFICATIONS|LICENSES & CERTIFICATIONS|INTERNSHIP COMPLETION CERTIFICATE)\b",
            r"(?:^|\n)\s*(ABOUT ME|SUMMARY|PROFILE|PROFESSIONAL SUMMARY|OBJECTIVE)\b",
            r"(?:^|\n)\s*(ACHIEVEMENTS|AWARDS|HONORS)\b",
            r"(?:^|\n)\s*(DECLARATION|TABLE OF CONTENTS|LEARNING OUTCOMES)\b",
            r"(?:^|\n)\s*(=== SECTION \d+:? [^=\n]+ ===)",
        ]
        combined = "|".join(section_patterns)
        matches = list(re.finditer(combined, text, flags=re.IGNORECASE))
        if not matches:
            return [{"section": "GENERAL", "text": text}]

        sections = []
        if matches[0].start() > 0:
            header_txt = text[:matches[0].start()].strip()
            if header_txt:
                sec_type = "PROJECTS" if re.search(r'(?:•|–|\bSystem:|\bAgent:|\bProject)', header_txt, re.IGNORECASE) else "HEADER"
                sections.append({"section": sec_type, "text": header_txt})

        for i, m in enumerate(matches):
            raw_title = m.group(0).strip(" =:\n\r")
            title = re.sub(r"^=== SECTION \d+:?\s*", "", raw_title, flags=re.IGNORECASE).strip()
            norm_title = title.upper()
            if re.search(r"CERTIF", norm_title):
                norm_title = "CERTIFICATIONS"
            elif re.search(r"SKILL", norm_title):
                norm_title = "SKILLS SUMMARY"
            elif re.search(r"PROJECT", norm_title):
                norm_title = "PROJECTS"
            elif re.search(r"EDUCAT|ACADEMIC", norm_title):
                norm_title = "EDUCATION"
            elif re.search(r"INTERN|EXPERIENCE", norm_title):
                norm_title = "INTERNSHIP EXPERIENCE"
            elif re.search(r"ABOUT|SUMMARY|PROFILE|OBJECTIVE", norm_title):
                norm_title = "ABOUT ME"
            elif re.search(r"ACHIEVE|AWARD|HONOR", norm_title):
                norm_title = "ACHIEVEMENTS"

            start = m.start()
            end = matches[i+1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            sections.append({
                "section": norm_title,
                "text": content
            })
        return sections

    def load_file(
        self,
        file_path: str,
        user_id: str = "default_user",
        document_id: Optional[str] = None
    ) -> List[Document]:
        """Loads and chunks a document with metadata (document_id, chunk_id, page, file_hash)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()
        supported_exts = {".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls"}
        if ext not in supported_exts:
            raise ValueError(f"Unsupported file format '{ext}'. Supported formats: PDF, DOCX, TXT, CSV, XLSX.")

        doc_id = document_id or str(uuid.uuid4())
        file_hash = self.calculate_file_hash(file_path)
        file_type = ext.lstrip(".")

        chunks: List[Document] = []

        # 1. PDF Documents: Extract Page-by-Page with Page Tracking
        if ext == ".pdf":
            from pypdf import PdfReader
            try:
                reader = PdfReader(file_path)
            except Exception as pe:
                raise ValueError(f"Failed to read PDF file '{filename}'. File may be corrupt or encrypted: {pe}")

            if len(reader.pages) == 0:
                raise ValueError(f"PDF '{filename}' contains no pages.")

            total_extracted_text = ""
            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                try:
                    raw_page_text = page.extract_text() or ""
                except Exception as ex:
                    logger.warning(f"Could not extract text from page {page_num} of {filename}: {ex}")
                    raw_page_text = ""

                clean_page = self.clean_text(raw_page_text)
                if not clean_page:
                    continue
                total_extracted_text += clean_page + "\n"

                # Check if page has semantic section headers
                page_sections = self._extract_sections(clean_page)
                for sec in page_sections:
                    sec_name = sec["section"]
                    sec_text = sec["text"]

                    if len(sec_text) <= self.chunk_size:
                        chunks.append(Document(
                            page_content=sec_text,
                            metadata={
                                "document_id": doc_id,
                                "source": filename,
                                "filename": filename,
                                "file_type": file_type,
                                "file_path": file_path,
                                "page": page_num,
                                "section": sec_name,
                                "user_id": user_id,
                                "file_hash": file_hash
                            }
                        ))
                    else:
                        sub_texts = self.splitter.split_text(sec_text)
                        for st in sub_texts:
                            prefix = f"[{sec_name}]\n" if not st.strip().startswith(sec_name) else ""
                            chunks.append(Document(
                                page_content=f"{prefix}{st}".strip(),
                                metadata={
                                    "document_id": doc_id,
                                    "source": filename,
                                    "filename": filename,
                                    "file_type": file_type,
                                    "file_path": file_path,
                                    "page": page_num,
                                    "section": sec_name,
                                    "user_id": user_id,
                                    "file_hash": file_hash
                                }
                            ))

            if not total_extracted_text.strip():
                raise ValueError(f"PDF document '{filename}' contains no extractable text.")

        # 2. DOCX Documents
        elif ext == ".docx":
            import docx
            try:
                doc = docx.Document(file_path)
            except Exception as de:
                raise ValueError(f"Failed to read DOCX file '{filename}': {de}")

            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            raw_text = "\n\n".join(paragraphs)
            clean_docx = self.clean_text(raw_text)
            if not clean_docx:
                raise ValueError(f"DOCX document '{filename}' contains no extractable text.")

            sections = self._extract_sections(clean_docx)
            for sec in sections:
                sec_name = sec["section"]
                sec_text = sec["text"]
                sub_texts = self.splitter.split_text(sec_text)
                for st in sub_texts:
                    chunks.append(Document(
                        page_content=st.strip(),
                        metadata={
                            "document_id": doc_id,
                            "source": filename,
                            "filename": filename,
                            "file_type": file_type,
                            "file_path": file_path,
                            "page": 1,
                            "section": sec_name,
                            "user_id": user_id,
                            "file_hash": file_hash
                        }
                    ))

        # 3. Plain Text Documents
        elif ext == ".txt":
            raw_text = ""
            for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        raw_text = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            clean_txt = self.clean_text(raw_text)
            if not clean_txt:
                raise ValueError(f"Text document '{filename}' is empty.")

            sections = self._extract_sections(clean_txt)
            for sec in sections:
                sec_name = sec["section"]
                sec_text = sec["text"]
                sub_texts = self.splitter.split_text(sec_text)
                for st in sub_texts:
                    chunks.append(Document(
                        page_content=st.strip(),
                        metadata={
                            "document_id": doc_id,
                            "source": filename,
                            "filename": filename,
                            "file_type": file_type,
                            "file_path": file_path,
                            "page": 1,
                            "section": sec_name,
                            "user_id": user_id,
                            "file_hash": file_hash
                        }
                    ))

        # 4. CSV Tabular Documents
        elif ext == ".csv":
            import pandas as pd
            try:
                df = pd.read_csv(file_path)
            except Exception as ce:
                raise ValueError(f"Failed to read CSV '{filename}': {ce}")

            if df.empty:
                raise ValueError(f"CSV document '{filename}' contains no rows.")

            csv_text = df.to_string()
            sub_texts = self.splitter.split_text(csv_text)
            for st in sub_texts:
                chunks.append(Document(
                    page_content=st.strip(),
                    metadata={
                        "document_id": doc_id,
                        "source": filename,
                        "filename": filename,
                        "file_type": file_type,
                        "file_path": file_path,
                        "page": 1,
                        "section": "TABLE",
                        "user_id": user_id,
                        "file_hash": file_hash
                    }
                ))

        # 5. Excel Documents (.xlsx, .xls)
        elif ext in [".xlsx", ".xls"]:
            import pandas as pd
            try:
                excel_sheets = pd.read_excel(file_path, sheet_name=None)
            except Exception as ee:
                raise ValueError(f"Failed to read Excel workbook '{filename}': {ee}")

            if not excel_sheets:
                raise ValueError(f"Excel workbook '{filename}' has no sheets.")

            has_data = False
            for sheet_idx, (sheet_name, df) in enumerate(excel_sheets.items()):
                if df.empty:
                    continue
                has_data = True
                sheet_text = f"--- Sheet: {sheet_name} ---\n" + df.to_string()
                sub_texts = self.splitter.split_text(sheet_text)
                for st in sub_texts:
                    chunks.append(Document(
                        page_content=st.strip(),
                        metadata={
                            "document_id": doc_id,
                            "source": filename,
                            "filename": filename,
                            "file_type": file_type,
                            "file_path": file_path,
                            "page": sheet_idx + 1,
                            "sheet_name": sheet_name,
                            "section": "TABLE",
                            "user_id": user_id,
                            "file_hash": file_hash
                        }
                    ))

            if not has_data:
                raise ValueError(f"Excel workbook '{filename}' contains no data.")

        # Assign unique chunk_id and total chunk counts
        total_chunks = len(chunks)
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{doc_id}_{idx}"
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = total_chunks

        logger.info(f"Loaded '{filename}' into {total_chunks} chunks (doc_id={doc_id})")
        return chunks

ingestion_service = DocumentIngestionService()
