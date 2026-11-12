from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from backend.app.core.llm import llm_service

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.llm = llm_service
        self.logger = logging.getLogger(f"agent.{name}")

    @abstractmethod
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent logic against the current workflow state."""
        pass
