import os
import re
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

class RAGVectorStore:
    """Persistent ChromaDB Vector Store for knowledge retrieval."""

    class IngestionResult(int):
        """An int subclass representing chunk count that also acts as a dictionary containing ingestion metadata."""
        def __new__(cls, val: int, details: Optional[Dict[str, Any]] = None):
            obj = super().__new__(cls, val)
            obj.details = details or {}
            return obj

        def get(self, key: str, default: Any = None) -> Any:
            return self.details.get(key, default)

        def __getitem__(self, key: str) -> Any:
            return self.details[key]

        def __contains__(self, key: str) -> bool:
            return key in self.details

        def items(self):
            return self.details.items()

        def keys(self):
            return self.details.keys()

        def values(self):
            return self.details.values()

    def __init__(self):
        self.persist_dir = settings.CHROMA_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL_NAME
        )
        self.collection_name = "knowledge_base"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )
        self.ensure_initial_documents()

    def ensure_initial_documents(self, force_reload: bool = False):
        """Pre-indexes knowledge documents from uploads folder into ChromaDB."""
        try:
            from backend.app.rag.ingestion import ingestion_service
            if not os.path.exists(settings.UPLOAD_DIR):
                return

            supported_exts = {".txt", ".pdf", ".docx", ".csv", ".xlsx", ".xls"}
            for fname in os.listdir(settings.UPLOAD_DIR):
                if fname.startswith("."):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext in supported_exts:
                    file_path = os.path.join(settings.UPLOAD_DIR, fname)
                    try:
                        file_hash = ingestion_service.calculate_file_hash(file_path)
                        existing = self.collection.get(where={"file_hash": file_hash})
                        if force_reload or not existing or not existing.get("ids"):
                            if existing and existing.get("ids"):
                                self.collection.delete(ids=existing["ids"])
                            # Also delete any older chunks with same filename if re-indexing
                            old_by_source = self.collection.get(where={"source": fname})
                            if old_by_source and old_by_source.get("ids"):
                                self.collection.delete(ids=old_by_source["ids"])

                            chunks = ingestion_service.load_file(file_path, user_id="default_user")
                            if chunks:
                                self.add_documents(chunks, force_reload=True)
                                logger.info(f"Pre-indexed {len(chunks)} chunks from {fname} into ChromaDB")
                    except Exception as fe:
                        logger.warning(f"Could not index document '{fname}': {fe}")
        except Exception as e:
            logger.warning(f"Could not initialize knowledge documents: {e}")

    def add_documents(self, documents: List[Any], force_reload: bool = False) -> IngestionResult:
        """Add LangChain Document objects to ChromaDB with duplicate check."""
        if not documents:
            return self.IngestionResult(0, {"chunks_added": 0, "already_indexed": False, "document_id": None})

        first_doc = documents[0]
        doc_id = first_doc.metadata.get("document_id")
        file_hash = first_doc.metadata.get("file_hash")
        source = first_doc.metadata.get("source")

        # Duplicate detection by file_hash
        if file_hash and not force_reload:
            existing = self.collection.get(where={"file_hash": file_hash})
            if existing and existing.get("ids") and len(existing["ids"]) > 0:
                logger.info(f"Document '{source}' (hash={file_hash[:8]}) already indexed ({len(existing['ids'])} chunks).")
                return self.IngestionResult(len(existing["ids"]), {
                    "chunks_added": len(existing["ids"]),
                    "already_indexed": True,
                    "document_id": existing["metadatas"][0].get("document_id", doc_id),
                    "message": "Document already indexed."
                })

        ids = []
        texts = []
        metadatas = []

        for idx, doc in enumerate(documents):
            chunk_id = doc.metadata.get("chunk_id") or f"{doc_id}_{idx}"
            ids.append(chunk_id)
            texts.append(doc.page_content)
            # Ensure metadata values are valid Chroma primitives (str, int, float, bool)
            clean_meta = {}
            for k, v in doc.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            # Guarantee page is an integer
            try:
                clean_meta["page"] = int(clean_meta.get("page", 1))
            except (ValueError, TypeError):
                clean_meta["page"] = 1
            metadatas.append(clean_meta)

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        return self.IngestionResult(len(texts), {
            "chunks_added": len(texts),
            "already_indexed": False,
            "document_id": doc_id,
            "message": f"Successfully indexed {len(texts)} chunks."
        })

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
        source: Optional[str] = None,
        document_id: Optional[str] = None,
        section: Optional[str] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search returning top matching passages with source citations, page numbers, and similarity scores."""
        # Normalize source to clean filename
        clean_source = os.path.basename(source) if source else None

        # Construct filter
        filter_clauses = []
        if document_id:
            filter_clauses.append({"document_id": document_id})
        if clean_source:
            filter_clauses.append({"source": clean_source})
        if user_id:
            filter_clauses.append({"user_id": user_id})

        where_filter = None
        if len(filter_clauses) == 1:
            where_filter = filter_clauses[0]
        elif len(filter_clauses) > 1:
            where_filter = {"$and": filter_clauses}

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(top_k * 2, max(self.count(), 1)),
                where=where_filter
            )
            # If no matches found with user_id, but a specific source or document_id was requested, retry without user_id
            if (not results or not results.get("documents") or not results["documents"][0]) and user_id and (clean_source or document_id):
                fallback_clauses = []
                if document_id:
                    fallback_clauses.append({"document_id": document_id})
                if clean_source:
                    fallback_clauses.append({"source": clean_source})
                fallback_filter = fallback_clauses[0] if len(fallback_clauses) == 1 else {"$and": fallback_clauses}
                logger.info(f"Retrying ChromaDB query for source/doc without user_id restriction: {fallback_filter}")
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=min(top_k * 2, max(self.count(), 1)),
                    where=fallback_filter
                )
        except Exception as e:
            logger.warning(f"ChromaDB filtered query failed: {e}")
            if not clean_source and not document_id:
                # Only fall back to global search if no specific document was targeted
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=min(top_k * 2, max(self.count(), 1))
                )
            else:
                results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        matched_items = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for doc_text, meta, dist in zip(docs, metas, distances):
                score = round(1.0 / (1.0 + float(dist)), 4)
                # Preserve page number from chunk metadata
                page_num = meta.get("page")
                if page_num is None:
                    page_num = meta.get("chunk_index", 0) + 1

                matched_items.append({
                    "content": doc_text,
                    "metadata": meta,
                    "score": score,
                    "distance": dist,
                    "source": meta.get("source", meta.get("filename", "unknown")),
                    "filename": meta.get("filename", meta.get("source", "unknown")),
                    "section": meta.get("section", "GENERAL"),
                    "page": page_num,
                    "document_id": meta.get("document_id", ""),
                    "chunk_id": meta.get("chunk_id", "")
                })

        # Determine target section: either explicitly passed or inferred from query intent
        target_section = section
        required_keywords = []

        if not target_section:
            q_lower = query_text.lower()
            if re.search(r'\b(certif\w*|license\w*)\b', q_lower):
                target_section = "CERTIFICATIONS"
                required_keywords = ["certif", "license"]
            elif re.search(r'\b(skill\w*|technolog\w*|tech stack|programming language\w*)\b', q_lower) and not re.search(r'\b(project|certif|intern)\b', q_lower):
                target_section = "SKILLS SUMMARY"
                required_keywords = ["skill", "technolog", "language", "framework"]
            elif re.search(r'\b(project\w*|system\w* built|app\w* built)\b', q_lower) and not re.search(r'\b(intern|certif)\b', q_lower):
                target_section = "PROJECTS"
                required_keywords = ["project", "developed", "built", "system"]
            elif re.search(r'\b(educat\w*|degree\w*|college|university|cgpa|gpa|b\.?tech)\b', q_lower):
                target_section = "EDUCATION"
                required_keywords = ["educat", "degree", "university", "college", "b.tech", "puc", "gpa", "cgpa"]
            elif re.search(r'\b(intern\w*|work experience)\b', q_lower):
                target_section = "INTERNSHIP EXPERIENCE"
                required_keywords = ["intern", "experience", "work"]
            elif re.search(r'\b(about me|bio|profile|objective)\b', q_lower) and not re.search(r'\b(skill|project|certif)\b', q_lower):
                target_section = "ABOUT ME"
                required_keywords = ["about", "profile", "summary"]

        section_items = []
        if target_section:
            try:
                where_sec = {"$and": [{"source": clean_source}, {"section": target_section}]} if clean_source else {"section": target_section}
                sec_records = self.collection.get(where=where_sec)
                if sec_records and sec_records.get("documents"):
                    seen_contents = set()
                    for s_doc, s_meta in zip(sec_records["documents"], sec_records["metadatas"]):
                        if s_doc not in seen_contents:
                            seen_contents.add(s_doc)
                            s_page = s_meta.get("page")
                            if s_page is None:
                                s_page = s_meta.get("chunk_index", 0) + 1
                            section_items.append({
                                "content": s_doc,
                                "metadata": s_meta,
                                "score": 0.95,
                                "distance": 0.05,
                                "source": s_meta.get("source", clean_source or "unknown"),
                                "filename": s_meta.get("filename", clean_source or "unknown"),
                                "section": s_meta.get("section", target_section),
                                "page": s_page,
                                "document_id": s_meta.get("document_id", ""),
                                "chunk_id": s_meta.get("chunk_id", "")
                            })
                    section_items.sort(key=lambda x: x.get("page", 0))
            except Exception as se:
                logger.debug(f"Direct section retrieval fallback: {se}")

        # If a specific section was targeted and section items were found:
        if target_section and section_items:
            seen_section_docs = {s["content"] for s in section_items}
            # Only include additional chunks from matched_items if they explicitly mention required keywords
            keyword_items = []
            if required_keywords:
                for m in matched_items:
                    if m["content"] not in seen_section_docs:
                        c_lower = m["content"].lower()
                        if any(re.search(r'\b' + kw, c_lower) for kw in required_keywords):
                            m["score"] = max(m.get("score", 0), 0.85)
                            keyword_items.append(m)
            matched_items = section_items + keyword_items
        elif section_items:
            seen_section_docs = {s["content"] for s in section_items}
            other_items = [m for m in matched_items if m["content"] not in seen_section_docs]
            matched_items = section_items + other_items

        # Apply similarity threshold if specified
        if similarity_threshold is not None:
            matched_items = [m for m in matched_items if m["score"] >= similarity_threshold]

        limit = max(top_k, len(section_items)) if section_items else top_k
        return matched_items[:limit]

    def retrieve_relevant_documents(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
        source: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Clean reusable retrieval function adhering to production RAG specification."""
        return self.query(
            query_text=query,
            top_k=top_k,
            document_id=document_id,
            source=source,
            similarity_threshold=threshold
        )

    def get_document_chunks(
        self,
        source: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves all indexed chunks for a document, ordered sequentially by chunk_index and page."""
        where_clauses = []
        if document_id:
            where_clauses.append({"document_id": document_id})
        if source:
            where_clauses.append({"source": source})

        if not where_clauses:
            return []

        where_filter = where_clauses[0] if len(where_clauses) == 1 else {"$and": where_clauses}
        try:
            records = self.collection.get(where=where_filter)
            if (not records or not records.get("documents")) and source and len(where_clauses) > 1:
                records = self.collection.get(where={"source": source})

            if not records or not records.get("documents"):
                return []

            chunks = []
            docs = records["documents"]
            metas = records["metadatas"] if records.get("metadatas") else [{}] * len(docs)
            for doc_text, meta in zip(docs, metas):
                chunk_idx = meta.get("chunk_index", 0)
                page_num = meta.get("page")
                if page_num is None:
                    page_num = chunk_idx + 1
                chunks.append({
                    "content": doc_text,
                    "metadata": meta,
                    "source": meta.get("source", source or "unknown"),
                    "filename": meta.get("filename", source or "unknown"),
                    "section": meta.get("section", "GENERAL"),
                    "page": page_num,
                    "chunk_index": chunk_idx,
                    "document_id": meta.get("document_id", document_id or ""),
                    "chunk_id": meta.get("chunk_id", "")
                })

            # Sort sequentially by chunk_index
            chunks.sort(key=lambda x: x.get("chunk_index", 0))
            return chunks
        except Exception as e:
            logger.error(f"Error retrieving full document chunks: {e}")
            return []

    def delete_document(self, document_id: str) -> int:
        """Deletes all chunks associated with a specific document_id."""
        try:
            records = self.collection.get(where={"document_id": document_id})
            if records and records.get("ids"):
                ids = records["ids"]
                self.collection.delete(ids=ids)
                logger.info(f"Deleted document {document_id} ({len(ids)} chunks removed).")
                return len(ids)
            return 0
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            raise

    def delete_by_source(self, source: str) -> int:
        """Deletes all chunks associated with a specific filename/source."""
        try:
            clean_source = os.path.basename(source)
            records = self.collection.get(where={"source": clean_source})
            if records and records.get("ids"):
                ids = records["ids"]
                self.collection.delete(ids=ids)
                logger.info(f"Deleted source '{clean_source}' ({len(ids)} chunks removed).")
                return len(ids)
            return 0
        except Exception as e:
            logger.error(f"Error deleting source {source}: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on the vector store including chunk count and unique documents."""
        try:
            total_chunks = self.collection.count()
            records = self.collection.get()
            docs_map = {}
            if records and records.get("metadatas"):
                for m in records["metadatas"]:
                    source = m.get("source") or m.get("filename") or "unknown"
                    doc_id = m.get("document_id") or ""
                    # Key by source/filename so all unique documents are properly reported
                    if source not in docs_map:
                        docs_map[source] = {
                            "document_id": doc_id,
                            "filename": source,
                            "file_type": m.get("file_type") or os.path.splitext(source)[1].lstrip("."),
                            "chunks_count": 0,
                            "file_hash": m.get("file_hash", "")
                        }
                    docs_map[source]["chunks_count"] += 1
                    if doc_id and not docs_map[source]["document_id"]:
                        docs_map[source]["document_id"] = doc_id

            sorted_docs = sorted(list(docs_map.values()), key=lambda x: x["filename"].lower())
            return {
                "status": "ready",
                "collection_name": self.collection_name,
                "total_chunks_indexed": total_chunks,
                "total_documents": len(docs_map),
                "embedding_model": settings.EMBEDDING_MODEL_NAME,
                "documents": sorted_docs
            }
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {
                "status": "error",
                "error": str(e),
                "total_chunks_indexed": self.collection.count() if hasattr(self, "collection") else 0,
                "documents": []
            }

    def count(self) -> int:
        return self.collection.count()

    def clear(self):
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )

rag_store = RAGVectorStore()
