from fastapi import APIRouter
from backend.app.api.endpoints_tasks import router as tasks_router
from backend.app.api.endpoints_rag import router as rag_router
from backend.app.api.endpoints_memory import router as memory_router
from backend.app.api.endpoints_schedule import router as schedule_router
from backend.app.api.endpoints_chat import router as chat_router
from backend.app.api.endpoints_auth import router as auth_router

api_router = APIRouter()

api_router.include_router(auth_router, prefix="/auth", tags=["Authentication & User Management"])
api_router.include_router(tasks_router, prefix="/tasks", tags=["Tasks & Workflow Execution"])

api_router.include_router(chat_router, prefix="/chat", tags=["Chat Memory & Conversation History"])
api_router.include_router(rag_router, prefix="/documents", tags=["RAG & Knowledge Base"])
api_router.include_router(rag_router, prefix="/rag", tags=["RAG & Knowledge Base"])
api_router.include_router(memory_router, prefix="/memory", tags=["User Memory & Personalization"])
api_router.include_router(schedule_router, prefix="/schedule", tags=["Scheduling & Automation"])
