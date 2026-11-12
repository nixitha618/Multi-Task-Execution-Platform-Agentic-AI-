from typing import Dict, Any, Callable, List, Optional
import logging
from backend.app.tools.calculator import CalculatorTool
from backend.app.tools.search import SearchTool
from backend.app.tools.data_analysis import DataAnalysisTool
from backend.app.tools.file_gen import FileGenTool
from backend.app.tools.email_tool import EmailTool
from backend.app.tools.rag import RAGTool
from backend.app.models.schemas import ToolExecutionResult

logger = logging.getLogger(__name__)

class ToolRegistry:
    """Central registry and dispatcher for all system tools."""

    def __init__(self):
        self._tools: Dict[str, Callable[..., ToolExecutionResult]] = {
            "calculator": CalculatorTool.execute,
            "search": SearchTool.execute,
            "data_analysis": DataAnalysisTool.execute,
            "file_gen": FileGenTool.execute,
            "email": EmailTool.execute,
            "rag": RAGTool.execute,
        }

    def register(self, name: str, func: Callable[..., ToolExecutionResult]):
        self._tools[name] = func

    def get_tool(self, name: str) -> Optional[Callable[..., ToolExecutionResult]]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    async def aexecute(self, tool_name: str, **kwargs) -> ToolExecutionResult:
        """Execute a tool asynchronously by name."""
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found in registry. Available tools: {self.list_tools()}"
            )

        try:
            import inspect
            # Defensive parameter inspection: verify signature accepts kwargs or filter to valid params
            sig = inspect.signature(tool)
            has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not has_var_keyword:
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            else:
                filtered_kwargs = kwargs

            # Synchronous tools are executed directly
            return tool(**filtered_kwargs)
        except TypeError as e:
            logger.warning(f"Parameter mismatch for tool '{tool_name}': {e}. Retrying with empty kwargs fallback.")
            try:
                return tool()
            except Exception:
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    result=None,
                    error=f"Parameter mismatch when invoking tool '{tool_name}': {str(e)}"
                )
        except Exception as e:
            logger.error(f"Error during tool '{tool_name}' execution: {e}", exc_info=True)
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e)
            )

tool_registry = ToolRegistry()
