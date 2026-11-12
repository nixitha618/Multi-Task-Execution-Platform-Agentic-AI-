import os
import shutil
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
import uuid
from backend.app.core.datetime_utils import get_ist_iso, get_ist_now
from backend.app.core.config import settings
from backend.app.rag.ingestion import ingestion_service
from backend.app.rag.vectorstore import rag_store
from backend.app.services.memory_service import memory_service
from backend.app.core.llm import llm_service
from backend.app.models.schemas import RAGChatRequest, RAGChatResponse, RAGChatSource

logger = logging.getLogger(__name__)
router = APIRouter()

class RAGQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    document_id: Optional[str] = None
    source: Optional[str] = None
    threshold: Optional[float] = None
    user_id: Optional[str] = "default_user"

class DocumentUploadResponse(BaseModel):
    success: bool
    filename: str
    file_path: str
    document_id: Optional[str] = None
    chunks_indexed: int
    message: str

class RAGQueryResponse(BaseModel):
    query: str
    results_count: int
    matches: List[Dict[str, Any]]

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form("default_user")
):
    """Upload and index a document (PDF, TXT, DOCX, CSV, XLSX) into ChromaDB RAG store."""
    try:
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Ingest and chunk document
        chunks = ingestion_service.load_file(file_path, user_id=user_id)
        if not chunks:
            return DocumentUploadResponse(
                success=False,
                filename=file.filename,
                file_path=file_path,
                document_id=None,
                chunks_indexed=0,
                message="File was empty or no extractable text found."
            )

        doc_id = chunks[0].metadata.get("document_id") if chunks and hasattr(chunks[0], "metadata") else None

        # Clean up any existing chunks for this file to prevent duplicate chunks in vector store
        rag_store.delete_by_source(file.filename)

        # Add to ChromaDB vector store
        indexed_count = rag_store.add_documents(chunks)

        return DocumentUploadResponse(
            success=True,
            filename=file.filename,
            file_path=file_path,
            document_id=doc_id,
            chunks_indexed=indexed_count,
            message=f"Successfully indexed {indexed_count} chunks from '{file.filename}' into RAG vector store."
        )
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@router.post("/query", response_model=RAGQueryResponse)
async def query_knowledge_base(
    body: Optional[RAGQueryRequest] = None,
    query: Optional[str] = Query(None, description="Query string if sent as URL parameter"),
    top_k: int = Query(5, description="Number of matches to return"),
    document_id: Optional[str] = Query(None, description="Filter by document_id"),
    source: Optional[str] = Query(None, description="Filter by filename/source"),
    threshold: Optional[float] = Query(None, description="Minimum similarity threshold")
):
    """Query ChromaDB vector store for semantic matches, citations, and scores.
    Supports both JSON body and query parameters."""
    try:
        active_query = (body.query if body and body.query else query)
        if not active_query:
            raise HTTPException(status_code=400, detail="Query parameter or JSON body with 'query' field is required.")

        k = (body.top_k if body and body.top_k is not None else top_k) or 5
        doc_id = body.document_id if body and body.document_id else document_id
        src = body.source if body and body.source else source
        th = body.threshold if body and body.threshold is not None else threshold

        results = rag_store.retrieve_relevant_documents(
            query=active_query,
            top_k=k,
            document_id=doc_id,
            source=src,
            threshold=th
        )

        return RAGQueryResponse(
            query=active_query,
            results_count=len(results),
            matches=results
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying RAG store: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
@router.get("/stats")
async def get_rag_status():
    """Get vector store statistics, total chunks, and indexed documents list."""
    try:
        return rag_store.get_stats()
    except Exception as e:
        logger.error(f"Error fetching RAG stats: {e}")
        return {
            "status": "error",
            "error": str(e),
            "total_chunks_indexed": rag_store.count() if hasattr(rag_store, "count") else 0,
            "documents": []
        }

@router.delete("/clear")
async def clear_knowledge_base():
    """Clear all indexed documents in the vector store."""
    try:
        rag_store.clear()
        return {"success": True, "message": "Knowledge base collection cleared."}
    except Exception as e:
        logger.error(f"Error clearing vector store: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/source/{filename:path}")
async def delete_document_by_source(filename: str):
    """Delete all chunks associated with a specific filename/source."""
    try:
        deleted = rag_store.delete_by_source(filename)
        return {
            "success": True,
            "filename": filename,
            "deleted_chunks": deleted,
            "message": f"Successfully deleted {deleted} chunks for '{filename}'."
        }
    except Exception as e:
        logger.error(f"Error deleting document by source '{filename}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{document_id}")
async def delete_document_by_id(document_id: str):
    """Delete all chunks associated with a specific document_id."""
    try:
        deleted = rag_store.delete_document(document_id)
        return {
            "success": True,
            "document_id": document_id,
            "deleted_chunks": deleted,
            "message": f"Successfully deleted {deleted} chunks for document {document_id}."
        }
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================================
# CONVERSATIONAL RAG CHATBOT ENDPOINTS
# =========================================================================

@router.post("/chat", response_model=RAGChatResponse)
async def chat_with_documents(req: RAGChatRequest):
    """
    Interactive Conversational RAG Chatbot Endpoint.
    Maintains multi-turn conversation memory, resolves follow-up references
    prior to previous questions, performs scoped vector search, and generates
    grounded responses with document citations.
    """
    try:
        session_id = req.session_id or f"rag_sess_{uuid.uuid4().hex[:12]}"
        user_id = req.user_id or "default_user"
        message = req.message.strip()
        doc_scope = req.document_scope.strip() if req.document_scope and req.document_scope.strip() else None
        top_k = req.top_k or 5

        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        # 1. Fetch prior conversation history before saving current message
        prior_messages = await memory_service.get_session_messages(session_id)

        # 2. Save current user question to persistent DB
        user_msg = await memory_service.save_message(
            session_id=session_id,
            role="user",
            content=message,
            metadata={"user_id": user_id, "document_scope": doc_scope}
        )

        # 3. Contextualize query if prior turns exist
        search_query = message
        history_text = ""
        if prior_messages:
            conv_lines = []
            for m in prior_messages[-6:]:
                role_label = "User" if m.get("role") == "user" else "Assistant"
                conv_lines.append(f"{role_label}: {m.get('content', '')[:300]}")
            history_text = "\n".join(conv_lines)

            # Check if current question has reference pronouns or follow-up phrasing
            followup_indicators = [
                "previous", "prior", "last", "earlier", "second", "third", "first",
                "it", "they", "them", "that", "those", "this", "these", "elaborate",
                "tell me more", "explain more", "what about", "and", "why", "how so"
            ]
            import re
            has_followup_clue = any(re.search(r"\b" + re.escape(w) + r"\b", message.lower()) for w in followup_indicators)

            if has_followup_clue or len(message.split()) <= 6:
                try:
                    contextualize_prompt = (
                        "Given the prior conversation history and a follow-up user question, reformulate the user question into a standalone, keyword-rich search query for document vector retrieval.\n\n"
                        f"Conversation History:\n{history_text}\n\n"
                        f"Follow-up Question: {message}\n\n"
                        "Standalone Search Query (concise, no punctuation or filler, resolving all pronouns):"
                    )
                    rewritten = await llm_service.ainvoke_text(
                        prompt=contextualize_prompt,
                        system_prompt="You reformulate follow-up questions into standalone search queries based on chat context. Output ONLY the search query string.",
                        fast=True
                    )
                    clean_rewritten = rewritten.strip().strip('"').strip("'")
                    if clean_rewritten and len(clean_rewritten) > 3 and "standalone search query" not in clean_rewritten.lower():
                        search_query = clean_rewritten
                        logger.info(f"Contextualized RAG query: '{message}' -> '{search_query}'")
                except Exception as ex:
                    logger.warning(f"Could not contextualize query, using raw query: {ex}")

        # 4. Perform vector search in ChromaDB
        chunks = []
        try:
            chunks = rag_store.query(
                query_text=search_query,
                top_k=top_k,
                source=doc_scope
            )
            # If no chunks found under rewritten query and it differed, fall back to raw message
            if not chunks and search_query != message:
                chunks = rag_store.query(
                    query_text=message,
                    top_k=top_k,
                    source=doc_scope
                )
        except Exception as q_err:
            logger.warning(f"ChromaDB search error: {q_err}")

        # 5. Extract citations, source objects, and context strings
        citations = []
        sources_list = []
        seen_citations = set()
        context_texts = []

        for c in chunks:
            content = c.get("content", "")
            context_texts.append(content)
            src = c.get("source") or c.get("filename") or "Document"
            page = c.get("page")
            score = c.get("score")
            cite_label = f"{src}" + (f" (Page {page})" if page is not None else "")
            if cite_label not in seen_citations:
                seen_citations.add(cite_label)
                citations.append(cite_label)
            sources_list.append(RAGChatSource(
                source=src,
                page=page,
                score=score,
                snippet=content[:250].strip()
            ))

        # 6. Retrieve user memory preferences
        user_prefs = await memory_service.get_user_preferences(user_id)
        pref_str = "\n".join([f"- {k}: {v}" for k, v in user_prefs.items()]) if user_prefs else ""

        # 7. Synthesize grounded conversational response
        system_prompt = (
            "You are an expert RAG Knowledge Assistant having an interactive conversation with the user.\n"
            "Guidelines:\n"
            "1. Base your answer strictly on the provided DOCUMENT CONTEXT and maintain continuous conversational context with the CONVERSATION HISTORY.\n"
            "2. When the user asks follow-up questions referencing previous answers or questions, maintain seamless continuity and directly address their intent.\n"
            "3. Cite the relevant document sources or section titles where appropriate.\n"
            "4. Use clear, well-structured Markdown formatting (bullet points, bold text, code blocks if applicable).\n"
            "5. If the document context does not contain enough information, explain what was found and politely clarify what is missing.\n"
            "6. Respect any user profile facts or formatting preferences provided."
        )

        synthesis_prompt = f"USER QUESTION:\n{message}\n\n"
        if history_text:
            synthesis_prompt += f"RECENT CONVERSATION HISTORY (Previous Questions & Answers):\n{history_text}\n\n"
        if pref_str:
            synthesis_prompt += f"USER PREFERENCES:\n{pref_str}\n\n"
        if context_texts:
            synthesis_prompt += "DOCUMENT CONTEXT FROM KNOWLEDGE BASE:\n" + "\n---\n".join(context_texts[:6]) + "\n\n"
        else:
            synthesis_prompt += "DOCUMENT CONTEXT FROM KNOWLEDGE BASE:\n[No direct document matches found in vector store]\n\n"
        synthesis_prompt += "Provide an accurate, grounded, and conversational response."

        answer = ""
        try:
            answer = await llm_service.ainvoke_text(
                prompt=synthesis_prompt,
                system_prompt=system_prompt,
                fast=False
            )
        except Exception as l_err:
            logger.warning(f"LLM synthesis invocation exception: {l_err}")

        if not answer or not answer.strip():
            if context_texts:
                bullets = "\n".join([f"- {c[:200].strip()}..." for c in context_texts[:4]])
                cites_str = ", ".join(citations) if citations else "Uploaded Documents"
                answer = f"Based on the indexed document context:\n\n{bullets}\n\n*Sources:* {cites_str}"
            else:
                answer = "I could not find relevant information in the uploaded documents to answer your question. Please ensure your files are indexed or try asking in another way."

        # 8. Save assistant response to persistent DB
        asst_msg = await memory_service.save_message(
            session_id=session_id,
            role="assistant",
            content=answer,
            metadata={"citations": citations, "document_scope": doc_scope}
        )

        return RAGChatResponse(
            session_id=session_id,
            message_id=asst_msg.get("id"),
            answer=answer,
            citations=citations,
            sources=sources_list,
            document_scope=doc_scope,
            timestamp=get_ist_iso()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in RAG chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/history/{session_id}")
async def get_rag_chat_history(session_id: str):
    """Retrieve message history for a specific RAG conversational session."""
    try:
        messages = await memory_service.get_session_messages(session_id)
        return {
            "session_id": session_id,
            "total_messages": len(messages),
            "messages": messages
        }
    except Exception as e:
        logger.error(f"Error retrieving RAG chat history for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/chat/history/{session_id}")
async def clear_rag_chat_history(session_id: str):
    """Clear message history for a RAG conversational session."""
    try:
        count = await memory_service.clear_session(session_id)
        return {
            "session_id": session_id,
            "cleared": True,
            "messages_deleted": count,
            "message": f"Cleared {count} messages from session {session_id}."
        }
    except Exception as e:
        logger.error(f"Error clearing RAG chat history for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
