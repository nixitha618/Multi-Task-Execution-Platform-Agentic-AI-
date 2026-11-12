import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from backend.app.core.datetime_utils import get_ist_now, get_ist_iso
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db, AsyncSessionLocal
from backend.app.models.db_models import Task, TaskStep, WorkflowLog
from backend.app.models.schemas import TaskCreateRequest, TaskResponse, TaskStepResponse
from backend.app.orchestration.graph import multi_agent_graph

import uuid
from backend.app.services.memory_service import memory_service

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory queue manager for active SSE streams
# task_id -> list of asyncio.Queue
active_streams: Dict[str, List[asyncio.Queue]] = {}

def get_task_queues(task_id: str) -> List[asyncio.Queue]:
    if task_id not in active_streams:
        active_streams[task_id] = []
    return active_streams[task_id]

async def broadcast_event(task_id: str, event: Dict[str, Any]):
    queues = get_task_queues(task_id)
    for q in queues:
        await q.put(event)

async def run_workflow_background(
    task_id: str,
    prompt: str,
    user_id: str,
    attached_files: List[str],
    session_id: Optional[str] = None,
    current_message_id: Optional[int] = None
):
    """Executes the full LangGraph multi-agent graph in the background and saves updates to DB."""
    async with AsyncSessionLocal() as session:
        stmt = select(Task).where(Task.id == task_id)
        task = (await session.execute(stmt)).scalar_one_or_none()
        if not task:
            return
        task.status = "UNDERSTANDING"
        await session.commit()

    initial_state = {
        "task_id": task_id,
        "user_id": user_id,
        "session_id": session_id,
        "current_message_id": current_message_id,
        "user_prompt": prompt,
        "attached_files": attached_files,
        "step_results": {},
        "artifacts": [],
        "retry_count": 0,
        "max_retries": 1,
        "events": []
    }

    try:
        # Stream intermediate graph updates
        last_event_count = 0
        final_state = None

        async for chunk in multi_agent_graph.astream(initial_state, stream_mode="values"):
            final_state = chunk
            events = chunk.get("events", [])
            # Broadcast any newly generated events
            if len(events) > last_event_count:
                for ev in events[last_event_count:]:
                    await broadcast_event(task_id, ev)
                    # Persist event as WorkflowLog in DB
                    async with AsyncSessionLocal() as session:
                        log_entry = WorkflowLog(
                            task_id=task_id,
                            level=ev.get("event_type", "INFO"),
                            agent_name=ev.get("agent_name"),
                            message=ev.get("message", ""),
                            metadata_json=ev.get("data")
                        )
                        session.add(log_entry)
                        await session.commit()
                last_event_count = len(events)

        # Update final task status in DB
        async with AsyncSessionLocal() as session:
            stmt = select(Task).where(Task.id == task_id)
            task = (await session.execute(stmt)).scalar_one_or_none()
            if task and final_state:
                task.status = "COMPLETED"
                task.task_type = final_state.get("task_spec", {}).get("task_type")
                task.plan = final_state.get("execution_plan")
                task.final_output = final_state.get("final_response")
                task.structured_data = {
                    "artifacts": final_state.get("artifacts", []),
                    "evaluation": final_state.get("evaluation", {}),
                    "step_results": final_state.get("step_results", {})
                }
                task.completed_at = get_ist_now()

                # Save individual TaskSteps to DB
                plan_steps = final_state.get("execution_plan", {}).get("steps", [])
                step_results = final_state.get("step_results", {})
                for s in plan_steps:
                    s_idx = str(s.get("step_index"))
                    s_res = step_results.get(s_idx, {})
                    step_rec = TaskStep(
                        task_id=task_id,
                        step_index=s.get("step_index"),
                        description=s.get("description"),
                        agent_name=s.get("agent_name"),
                        tool_name=s.get("tool_name"),
                        input_data=s.get("tool_input_template"),
                        output_data=s_res,
                        status="COMPLETED" if s_res.get("success", True) else "FAILED"
                    )
                    session.add(step_rec)

                await session.commit()

        # Save assistant message to persistent conversation_messages table
        if session_id and final_state and final_state.get("final_response"):
            try:
                await memory_service.save_message(
                    session_id=session_id,
                    role="assistant",
                    content=final_state.get("final_response"),
                    metadata={
                        "task_id": task_id,
                        "artifacts": final_state.get("artifacts", [])
                    }
                )
            except Exception as ex_mem:
                logger.warning(f"Failed to record assistant message in session history: {ex_mem}")

        # Send terminal SSE event with final output data
        await broadcast_event(task_id, {
            "timestamp": get_ist_iso(),
            "event_type": "DONE",
            "agent_name": "System",
            "message": "Task workflow finished.",
            "data": {
                "status": "COMPLETED",
                "final_output": final_state.get("final_response"),
                "artifacts": final_state.get("artifacts", [])
            }
        })

    except Exception as e:
        logger.error(f"Error in task {task_id}: {e}", exc_info=True)
        async with AsyncSessionLocal() as session:
            stmt = select(Task).where(Task.id == task_id)
            task = (await session.execute(stmt)).scalar_one_or_none()
            if task:
                task.status = "FAILED"
                task.error_message = str(e)
                await session.commit()

        await broadcast_event(task_id, {
            "timestamp": get_ist_iso(),
            "event_type": "ERROR",
            "agent_name": "System",
            "message": f"Task execution failed: {str(e)}"
        })

from sqlalchemy.orm import selectinload

@router.post("", response_model=TaskResponse)
async def create_task(
    req: TaskCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Submit a natural language prompt to start an autonomous multi-agent task."""
    session_id = req.session_id or str(uuid.uuid4())

    # Save user message to persistent conversation_messages table
    user_msg = await memory_service.save_message(
        session_id=session_id,
        role="user",
        content=req.prompt,
        metadata={"user_id": req.user_id or "default_user"}
    )
    current_message_id = user_msg.get("id")

    task = Task(
        user_id=req.user_id,
        session_id=session_id,
        prompt=req.prompt,
        status="PENDING"
    )
    db.add(task)
    await db.commit()

    background_tasks.add_task(
        run_workflow_background,
        task_id=task.id,
        prompt=req.prompt,
        user_id=req.user_id or "default_user",
        attached_files=req.attached_files or [],
        session_id=session_id,
        current_message_id=current_message_id
    )

    stmt = select(Task).options(selectinload(Task.steps)).where(Task.id == task.id)
    return (await db.execute(stmt)).scalar_one()

@router.get("", response_model=List[TaskResponse])
async def list_tasks(
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """Retrieve recent task execution history."""
    stmt = select(Task).options(selectinload(Task.steps)).order_by(desc(Task.created_at)).limit(limit)
    res = await db.execute(stmt)
    return res.scalars().all()

@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieve specific task details, plan, steps, and final response."""
    stmt = select(Task).options(selectinload(Task.steps)).where(Task.id == task_id)
    res = await db.execute(stmt)
    task = res.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.get("/{task_id}/stream")
async def stream_task_events(task_id: str):
    """Server-Sent Events (SSE) live endpoint streaming real-time agent thoughts and actions."""
    queue = asyncio.Queue()
    queues = get_task_queues(task_id)
    queues.append(queue)

    async def event_generator():
        try:
            # Replay any existing logs that occurred before client connected
            async with AsyncSessionLocal() as session:
                stmt = select(WorkflowLog).where(WorkflowLog.task_id == task_id).order_by(WorkflowLog.id)
                existing_logs = (await session.execute(stmt)).scalars().all()
                for log in existing_logs:
                    replayed_ev = {
                        "timestamp": get_ist_iso(),
                        "event_type": log.level,
                        "agent_name": log.agent_name,
                        "message": log.message,
                        "data": log.metadata_json or {}
                    }
                    yield f"data: {json.dumps(replayed_ev)}\n\n"

                # Check if task already finished
                stmt_t = select(Task).where(Task.id == task_id)
                task_rec = (await session.execute(stmt_t)).scalar_one_or_none()
                if task_rec and task_rec.status in ["COMPLETED", "FAILED"]:
                    done_ev = {
                        "timestamp": get_ist_iso(),
                        "event_type": "DONE" if task_rec.status == "COMPLETED" else "ERROR",
                        "agent_name": "System",
                        "message": "Task workflow finished.",
                        "data": {
                            "status": task_rec.status,
                            "final_output": task_rec.final_output,
                            "artifacts": (task_rec.structured_data or {}).get("artifacts", [])
                        }
                    }
                    yield f"data: {json.dumps(done_ev)}\n\n"
                    return

            while True:
                # Wait for next event or timeout check
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("event_type") in ["DONE", "ERROR"]:
                        break
                except asyncio.TimeoutError:
                    # Keep-alive heartbeat
                    yield ": heartbeat\n\n"
        finally:
            if queue in queues:
                queues.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
