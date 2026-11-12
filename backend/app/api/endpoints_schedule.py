from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.models.db_models import ScheduledJob
from backend.app.models.schemas import ScheduledJobCreateRequest, ScheduledJobResponse
from backend.app.services.scheduler import scheduler_service

router = APIRouter()

class TriggerJobRequest(BaseModel):
    attached_files: Optional[List[str]] = None

@router.get("", response_model=List[ScheduledJobResponse])
async def list_scheduled_jobs(db: AsyncSession = Depends(get_db)):
    """List all configured recurring automated jobs."""
    stmt = select(ScheduledJob).order_by(ScheduledJob.created_at)
    res = await db.execute(stmt)
    return res.scalars().all()

@router.post("", response_model=ScheduledJobResponse)
async def create_scheduled_job(req: ScheduledJobCreateRequest, db: AsyncSession = Depends(get_db)):
    """Register a new recurring autonomous AI workflow."""
    job = ScheduledJob(
        user_id=req.user_id,
        name=req.name,
        prompt=req.prompt,
        cron_expression=req.cron_expression,
        is_active=True
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Register in APScheduler
    scheduler_service.add_job(
        job_id=job.id,
        prompt=job.prompt,
        user_id=job.user_id or "default_user",
        cron_expression=job.cron_expression
    )

    return job

@router.delete("/{job_id}")
async def delete_scheduled_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """Stop and delete a scheduled job permanently."""
    stmt = select(ScheduledJob).where(ScheduledJob.id == job_id)
    job = (await db.execute(stmt)).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Scheduled job not found")

    # Remove from APScheduler
    scheduler_service.remove_job(job_id)

    # Delete from DB
    await db.delete(job)
    await db.commit()
    return {"status": "deleted", "id": job_id, "name": job.name}

@router.post("/{job_id}/toggle", response_model=ScheduledJobResponse)
@router.patch("/{job_id}/toggle", response_model=ScheduledJobResponse)
async def toggle_scheduled_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """Pause or resume a scheduled recurring job."""
    stmt = select(ScheduledJob).where(ScheduledJob.id == job_id)
    job = (await db.execute(stmt)).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Scheduled job not found")

    job.is_active = not job.is_active

    if job.is_active:
        # Resume/re-add to APScheduler
        scheduler_service.add_job(
            job_id=job.id,
            prompt=job.prompt,
            user_id=job.user_id or "default_user",
            cron_expression=job.cron_expression
        )
    else:
        # Pause/remove from APScheduler
        scheduler_service.remove_job(job.id)

    await db.commit()
    await db.refresh(job)
    return job

@router.post("/{job_id}/trigger")
async def trigger_scheduled_job_now(
    job_id: str,
    req: Optional[TriggerJobRequest] = None,
    db: AsyncSession = Depends(get_db)
):
    """Immediately trigger a scheduled workflow run in the background."""
    stmt = select(ScheduledJob).where(ScheduledJob.id == job_id)
    job = (await db.execute(stmt)).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Scheduled job not found")

    attached = req.attached_files if req else None

    import asyncio
    asyncio.create_task(
        scheduler_service.execute_scheduled_task(
            job_id=job.id,
            prompt=job.prompt,
            user_id=job.user_id or "default_user",
            attached_files=attached
        )
    )
    return {"status": "triggered", "id": job_id, "name": job.name}

