from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from backend.app.services.memory_service import memory_service
from backend.app.models.schemas import MemoryCreateRequest, MemoryResponse

router = APIRouter()

@router.get("", response_model=List[MemoryResponse])
async def list_memories(
    user_id: str = "default_user",
    category: Optional[str] = None,
    limit: int = 20
):
    """Retrieve user long-term and short-term memories and preferences."""
    return await memory_service.get_memories(user_id=user_id, category=category, limit=limit)

@router.post("", response_model=MemoryResponse)
async def create_memory(req: MemoryCreateRequest):
    """Store user preference, fact, or workflow rule."""
    mem = await memory_service.add_memory(
        user_id=req.user_id,
        category=req.category,
        key=req.key,
        value=req.value,
        importance=req.importance
    )
    return mem

@router.delete("/{memory_id}")
async def remove_memory(memory_id: str):
    """Delete a memory entry."""
    success = await memory_service.delete_memory(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory item not found")
    return {"message": "Memory deleted successfully"}
