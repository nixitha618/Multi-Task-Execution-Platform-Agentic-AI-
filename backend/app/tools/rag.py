import logging
from typing import Optional, Dict, Any, List
from backend.app.models.schemas import ToolExecutionResult
from backend.app.rag.vectorstore import rag_store

logger = logging.getLogger(__name__)

class RAGTool:
    """Tool for querying the ChromaDB knowledge base with semantic retrieval."""

    @staticmethod
    def execute(
        query: Optional[str] = None,
        top_k: int = 5,
        user_id: str = "default_user",
        source: Optional[str] = None,
        document_id: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs
    ) -> ToolExecutionResult:
        try:
            actual_query = query or kwargs.get("search_query") or kwargs.get("q") or kwargs.get("question") or ""
            results = rag_store.retrieve_relevant_documents(
                query=str(actual_query),
                top_k=top_k,
                document_id=document_id or kwargs.get("doc_id"),
                source=source or kwargs.get("file_path"),
                threshold=threshold
            )
            return ToolExecutionResult(
                tool_name="rag",
                success=True,
                result=results,
                error=None
            )
        except Exception as e:
            logger.error(f"Error executing RAGTool: {e}", exc_info=True)
            return ToolExecutionResult(
                tool_name="rag",
                success=False,
                result=None,
                error=str(e)
            )
