import uuid
from datetime import datetime
from backend.app.core.datetime_utils import get_ist_now
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from backend.app.core.database import Base

def generate_uuid() -> str:
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=True, index=True)
    password_hash = Column(String(255), nullable=True) # bcrypt salted hash
    full_name = Column(String(150), nullable=True)
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    preferences = Column(JSON, default=dict)
    created_at = Column(DateTime, default=get_ist_now)
    updated_at = Column(DateTime, default=get_ist_now, onupdate=get_ist_now)

    tasks = relationship("Task", back_populates="user", cascade="all, delete-orphan")
    memories = relationship("UserMemory", back_populates="user", cascade="all, delete-orphan")
    scheduled_jobs = relationship("ScheduledJob", back_populates="user", cascade="all, delete-orphan")

class Task(Base):
    __tablename__ = "tasks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    session_id = Column(String(64), nullable=True, index=True)
    prompt = Column(Text, nullable=False)
    task_type = Column(String(50), nullable=True) # calculation, data_analysis, rag_qa, report, automation, general
    status = Column(String(30), default="PENDING", index=True) # PENDING, UNDERSTANDING, PLANNING, EXECUTING, EVALUATING, COMPLETED, FAILED
    plan = Column(JSON, nullable=True) # Structured execution plan
    final_output = Column(Text, nullable=True) # Markdown formatted response
    structured_data = Column(JSON, nullable=True) # Artifacts, charts, tables
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=get_ist_now)
    updated_at = Column(DateTime, default=get_ist_now, onupdate=get_ist_now)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="tasks")
    steps = relationship("TaskStep", back_populates="task", cascade="all, delete-orphan", order_by="TaskStep.step_index")
    logs = relationship("WorkflowLog", back_populates="task", cascade="all, delete-orphan", order_by="WorkflowLog.created_at")

class TaskStep(Base):
    __tablename__ = "task_steps"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    task_id = Column(String(36), ForeignKey("tasks.id"), nullable=False, index=True)
    step_index = Column(Integer, nullable=False)
    description = Column(Text, nullable=False)
    agent_name = Column(String(50), nullable=False) # TaskUnderstanding, Planner, Research, Execution, Memory, Evaluator, Response
    tool_name = Column(String(50), nullable=True) # calculator, data_analysis, search, file_gen, email, rag
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    status = Column(String(30), default="PENDING") # PENDING, IN_PROGRESS, COMPLETED, FAILED, RETRYING
    duration_seconds = Column(Float, default=0.0)
    created_at = Column(DateTime, default=get_ist_now)

    task = relationship("Task", back_populates="steps")

class WorkflowLog(Base):
    __tablename__ = "workflow_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(36), ForeignKey("tasks.id"), nullable=True, index=True)
    level = Column(String(20), default="INFO") # INFO, WARNING, ERROR, DEBUG, AGENT_THOUGHT
    agent_name = Column(String(50), nullable=True)
    message = Column(Text, nullable=False)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=get_ist_now, index=True)

    task = relationship("Task", back_populates="logs")

class UserMemory(Base):
    __tablename__ = "user_memories"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    category = Column(String(50), default="general") # preference, fact, task_history, rule
    key = Column(String(100), nullable=False, index=True)
    value = Column(Text, nullable=False)
    importance = Column(Float, default=1.0)
    created_at = Column(DateTime, default=get_ist_now)
    updated_at = Column(DateTime, default=get_ist_now, onupdate=get_ist_now)

    user = relationship("User", back_populates="memories")

class ScheduledJob(Base):
    __tablename__ = "scheduled_jobs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    name = Column(String(100), nullable=False)
    prompt = Column(Text, nullable=False)
    cron_expression = Column(String(50), nullable=False) # e.g. '0 9 * * *'
    is_active = Column(Boolean, default=True)
    last_run = Column(DateTime, nullable=True)
    next_run = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=get_ist_now)

    user = relationship("User", back_populates="scheduled_jobs")

class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "user", "assistant", "system"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=get_ist_now, index=True)
    message_metadata = Column("metadata", JSON, default=dict, nullable=True)
