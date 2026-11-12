import uuid
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from backend.app.core.datetime_utils import get_ist_now
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from backend.app.core.database import get_db, AsyncSessionLocal
from backend.app.models.db_models import Task, TaskStep
from backend.app.models.schemas import (
    ChatMessageResponse,
    ChatSessionHistoryResponse,
    ChatSubmitRequest,
    ChatSubmitResponse,
    TaskCreateRequest
)
from backend.app.services.memory_service import memory_service
from backend.app.orchestration.graph import multi_agent_graph

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get('/history/{session_id}', response_model=ChatSessionHistoryResponse)
async def get_session_history(session_id: str):
    messages = await memory_service.get_session_messages(session_id)
    return ChatSessionHistoryResponse(
        session_id=session_id,
        total_messages=len(messages),
        messages=messages
    )

@router.delete('/history/{session_id}')
async def clear_session_history(session_id: str):
    await memory_service.clear_session(session_id)
    return {
        'status': 'CLEARED',
        'session_id': session_id,
        'message': f'Conversation history for session {session_id} has been cleared.'
    }

@router.post('', response_model=ChatSubmitResponse)
async def submit_chat_message(
    req: ChatSubmitRequest,
    db: AsyncSession = Depends(get_db)
):
    session_id = req.session_id or str(uuid.uuid4())
    user_id = req.user_id or 'default_user'
    prompt = req.message.strip()

    if not prompt:
        raise HTTPException(status_code=400, detail='Message cannot be empty')

    user_msg = await memory_service.save_message(
        session_id=session_id,
        role='user',
        content=prompt,
        metadata={'user_id': user_id}
    )
    current_message_id = user_msg.get('id')

    task = Task(
        user_id=user_id,
        session_id=session_id,
        prompt=prompt,
        status='UNDERSTANDING'
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    initial_state = {
        'task_id': task.id,
        'user_id': user_id,
        'session_id': session_id,
        'current_message_id': current_message_id,
        'user_prompt': prompt,
        'attached_files': req.attached_files or [],
        'step_results': {},
        'artifacts': [],
        'retry_count': 0,
        'max_retries': 1,
        'events': []
    }

    try:
        final_state = await multi_agent_graph.ainvoke(initial_state)

        final_output = final_state.get('final_response', 'Workflow completed without response.')
        artifacts = final_state.get('artifacts', [])
        
        sources = []
        for s in final_state.get('step_results', {}).values():
            if isinstance(s, dict) and s.get('citations'):
                sources.extend(s.get('citations', []))

        async with AsyncSessionLocal() as session:
            stmt = select(Task).where(Task.id == task.id)
            db_task = (await session.execute(stmt)).scalar_one_or_none()
            if db_task:
                db_task.status = 'COMPLETED'
                db_task.task_type = final_state.get('task_spec', {}).get('task_type')
                db_task.plan = final_state.get('execution_plan')
                db_task.final_output = final_output
                db_task.structured_data = {
                    'artifacts': artifacts,
                    'evaluation': final_state.get('evaluation', {}),
                    'step_results': final_state.get('step_results', {})
                }
                db_task.completed_at = get_ist_now()
                await session.commit()

        return ChatSubmitResponse(
            session_id=session_id,
            task_id=task.id,
            answer=final_output,
            artifacts=artifacts,
            sources=list(dict.fromkeys(sources))
        )

    except Exception as e:
        logger.error(f'Chat workflow error for task {task.id}: {e}', exc_info=True)
        async with AsyncSessionLocal() as session:
            stmt = select(Task).where(Task.id == task.id)
            db_task = (await session.execute(stmt)).scalar_one_or_none()
            if db_task:
                db_task.status = 'FAILED'
                db_task.error_message = str(e)
                await session.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Workflow execution failed: {str(e)}'
        )
