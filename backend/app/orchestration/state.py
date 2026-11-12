from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    task_id: str
    user_id: str
    session_id: Optional[str]
    current_message_id: Optional[int]
    user_prompt: str
    attached_files: List[str]
    memory_context: str
    conversation_context: str
    chat_history: List[Dict[str, Any]]
    deterministic_memory_result: Optional[Dict[str, Any]]
    task_spec: Dict[str, Any]
    execution_plan: Dict[str, Any]
    step_results: Dict[str, Any]
    artifacts: List[str]
    retry_count: int
    max_retries: int
    needs_retry: bool
    retry_instructions: Optional[str]
    evaluation: Dict[str, Any]
    final_response: str
    status: str
    events: List[Dict[str, Any]]

