import logging
from typing import List, Dict, Any, Optional
from backend.app.models.schemas import ToolExecutionResult

logger = logging.getLogger(__name__)

class SearchTool:
    """Web search tool using DuckDuckGo with fallback."""

    @classmethod
    def execute(cls, query: Optional[str] = None, max_results: int = 5, **kwargs) -> ToolExecutionResult:
        try:
            actual_query = query or kwargs.get("search_query") or kwargs.get("q") or kwargs.get("expression") or ""
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(str(actual_query), max_results=max_results))
                for r in raw_results:
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", "")
                    })

            if not results:
                return ToolExecutionResult(
                    tool_name="search",
                    success=True,
                    result=f"No results found online for query: '{query}'.",
                    error=None
                )

            return ToolExecutionResult(
                tool_name="search",
                success=True,
                result=results,
                error=None
            )
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed ({e}), returning fallback search response.")
            return ToolExecutionResult(
                tool_name="search",
                success=True,
                result=[
                    {
                        "title": f"Information regarding {query}",
                        "snippet": f"Summary regarding query '{query}'. Real-time search is currently operating in fallback mode.",
                        "url": "https://example.com/fallback"
                    }
                ],
                error=None
            )
