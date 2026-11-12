from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

# --- Task API Schemas ---
class TaskCreateRequest(BaseModel):
    prompt: str = Field(..., description="User prompt or natural language instruction")
    user_id: Optional[str] = Field("default_user", description="User identifier")
    session_id: Optional[str] = Field(None, description="Conversation session identifier")
    attached_files: Optional[List[str]] = Field(default_factory=list, description="Paths to uploaded files")

class TaskStepResponse(BaseModel):
    id: str
    step_index: int
    description: str
    agent_name: str
    tool_name: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    status: str
    duration_seconds: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class TaskResponse(BaseModel):
    id: str
    user_id: Optional[str]
    session_id: Optional[str] = None
    prompt: str
    task_type: Optional[str]
    status: str
    plan: Optional[Dict[str, Any]] = None
    final_output: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    steps: List[TaskStepResponse] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)

# --- Agent 1: Task Understanding Schemas ---
class TaskUnderstandingResult(BaseModel):
    intent: str = Field(..., description="Primary user intent")
    task_type: str = Field(..., description="e.g., summary, detailed_summary, key_points, extraction, table, report, explanation, comparison, question_answering, calculation, data_analysis, memory_query, general")
    objective: str = Field(..., description="Clear, concise single-sentence goal")
    entities: List[str] = Field(default_factory=list, description="Key entities, filenames, metrics extracted")
    required_tools: List[str] = Field(default_factory=list, description="Tools needed (calculator, search, data_analysis, file_gen, email, rag, memory)")
    complexity: str = Field("medium", description="simple, medium, complex")
    constraints: List[str] = Field(default_factory=list, description="Explicit constraints or format rules")
    document_ids: List[str] = Field(default_factory=list, description="Referenced or uploaded document names/IDs")
    output_format: str = Field("chat", description="chat, pdf, email, markdown")
    send_email: bool = Field(False, description="Whether output should be delivered via email")
    recipient: Optional[str] = Field(None, description="Target email recipient if send_email is True")
    detail_level: str = Field("medium", description="brief, medium, detailed")

# --- Agent 2: Planner Schemas ---
class PlanStep(BaseModel):
    step_index: int = Field(..., description="1-indexed step number")
    description: str = Field(..., description="Detailed action to take in this step")
    agent_name: str = Field(..., description="Assigned agent: Research, Execution, Memory")
    tool_name: Optional[str] = Field(None, description="Tool to use: calculator, search, data_analysis, file_gen, email, rag")
    tool_input_template: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Expected inputs for tool")
    depends_on: List[int] = Field(default_factory=list, description="Preceding step indices required before this step")

class ExecutionPlan(BaseModel):
    objective: str
    steps: List[PlanStep] = Field(..., min_length=1)
    estimated_steps: int

# --- Agent 4: Tool Calling Schemas ---
class ToolExecutionResult(BaseModel):
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    artifacts: List[str] = Field(default_factory=list, description="Paths to generated charts, files, reports")

# --- Agent 6: Evaluator / Critic Schemas ---
class EvaluationResult(BaseModel):
    is_satisfactory: bool = Field(..., description="True if results fully satisfy user request")
    score: float = Field(..., ge=0.0, le=1.0, description="Quality score between 0.0 and 1.0")
    feedback: str = Field(..., description="Detailed evaluation reasoning")
    issues: List[str] = Field(default_factory=list, description="Missing elements or inconsistencies detected")
    needs_retry: bool = Field(False, description="True if self-correction retry should be executed")
    retry_instructions: Optional[str] = Field(None, description="Concrete instructions to fix flaws in next retry")

# --- Agent 7: Response Schemas ---
class FinalResponse(BaseModel):
    text: str = Field(..., description="Rich markdown formatted final answer")
    artifacts: List[str] = Field(default_factory=list, description="List of generated output files/images")
    metadata: Dict[str, Any] = Field(default_factory=dict)

# --- Workflow Event / Live SSE Schema ---
class WorkflowEvent(BaseModel):
    task_id: str
    timestamp: str
    event_type: str # status_change, agent_start, agent_thought, tool_call, tool_result, step_complete, evaluation, task_complete, error
    agent_name: Optional[str] = None
    message: str
    data: Optional[Dict[str, Any]] = None

# --- Memory Schemas ---
class MemoryCreateRequest(BaseModel):
    user_id: str = "default_user"
    category: str = "general"
    key: str
    value: str
    importance: float = 1.0

class MemoryResponse(BaseModel):
    id: str
    user_id: str
    category: str
    key: str
    value: str
    importance: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

# --- Scheduled Job Schemas ---
class ScheduledJobCreateRequest(BaseModel):
    user_id: str = "default_user"
    name: str
    prompt: str
    cron_expression: str # e.g. "0 9 * * *"

class ScheduledJobResponse(BaseModel):
    id: str
    user_id: Optional[str]
    name: str
    prompt: str
    cron_expression: str
    is_active: bool
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

# --- Chat & Conversation History Schemas ---
class ChatMessageResponse(BaseModel):
    id: int
    session_id: str
    role: str
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)

class ChatSessionHistoryResponse(BaseModel):
    session_id: str
    total_messages: int
    messages: List[ChatMessageResponse] = Field(default_factory=list)

class ChatSubmitRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Conversation session identifier")
    message: str = Field(..., description="User message or instruction")
    user_id: Optional[str] = Field("default_user", description="User identifier")
    attached_files: Optional[List[str]] = Field(default_factory=list, description="Optional attached files")

class ChatSubmitResponse(BaseModel):
    session_id: str
    task_id: str
    answer: str
    artifacts: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

# --- Authentication Schemas ---
class UserSignUpRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="Password meeting security criteria")
    confirm_password: Optional[str] = Field(None, description="Password confirmation matching password")
    username: Optional[str] = Field(None, description="Optional chosen username")
    full_name: Optional[str] = Field(None, description="User's full name")

class UserLoginRequest(BaseModel):
    email_or_username: str = Field(..., description="Email address or username")
    password: str = Field(..., description="Account password")

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool = True
    last_login: Optional[datetime] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class AuthResponse(BaseModel):
    success: bool
    message: str
    user: Optional[UserResponse] = None
    token: Optional[str] = None

# --- Conversational RAG Chatbot Schemas ---
class RAGChatRequest(BaseModel):
    message: str = Field(..., description="User question or follow-up regarding indexed documents")
    session_id: Optional[str] = Field(None, description="Conversation session identifier for multi-turn context")
    user_id: Optional[str] = Field("default_user", description="User identifier")
    document_scope: Optional[str] = Field(None, description="Optional target document filename to restrict search")
    top_k: Optional[int] = Field(5, description="Number of knowledge chunks to retrieve")

class RAGChatSource(BaseModel):
    source: str = Field(..., description="Document source or filename")
    page: Optional[int] = Field(None, description="Page number if applicable")
    score: Optional[float] = Field(None, description="Similarity or relevance score")
    snippet: Optional[str] = Field(None, description="Snippet text from the chunk")

class RAGChatResponse(BaseModel):
    session_id: str
    message_id: Optional[int] = None
    answer: str
    citations: List[str] = Field(default_factory=list)
    sources: List[RAGChatSource] = Field(default_factory=list)
    document_scope: Optional[str] = None
    timestamp: str

