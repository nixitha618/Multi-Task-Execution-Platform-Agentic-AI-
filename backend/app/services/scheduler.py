import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select
from backend.app.core.datetime_utils import IST, get_ist_now

from backend.app.core.config import settings
from backend.app.core.database import AsyncSessionLocal
from backend.app.models.db_models import ScheduledJob, Task
from backend.app.orchestration.graph import multi_agent_graph

logger = logging.getLogger(__name__)

class TaskSchedulerService:
    """Automates scheduled task execution (daily reports, recurring analyses)."""

    def __init__(self):
        self.scheduler = AsyncIOScheduler(timezone=IST)
        self.is_running = False

    def start(self):
        if not self.is_running:
            self.scheduler.start()
            self.is_running = True
            logger.info("Task Scheduler Service started.")

    def shutdown(self):
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("Task Scheduler Service shut down.")

    async def load_jobs_from_db(self):
        try:
            async with AsyncSessionLocal() as session:
                stmt = select(ScheduledJob).where(ScheduledJob.is_active == True)
                jobs = (await session.execute(stmt)).scalars().all()
                for job in jobs:
                    self.add_job(
                        job_id=job.id,
                        prompt=job.prompt,
                        user_id=job.user_id or "default_user",
                        cron_expression=job.cron_expression
                    )
            logger.info(f"Loaded {len(jobs)} active scheduled jobs from database into scheduler.")
        except Exception as e:
            logger.warning(f"Could not load scheduled jobs from DB: {e}")

    async def execute_scheduled_task(
        self,
        job_id: str,
        prompt: str,
        user_id: str,
        attached_files: Optional[List[str]] = None
    ):
        clean_prompt = prompt.strip().strip('"\'')
        logger.info(f"Triggering scheduled job {job_id}: '{clean_prompt}'")

        active_files = [os.path.basename(f) for f in (attached_files or []) if f]
        from backend.app.services.report_builder import should_auto_associate_upload
        if not active_files and should_auto_associate_upload(clean_prompt, attached_files) and os.path.exists(settings.UPLOAD_DIR):
            recent_uploads = sorted(
                [
                    f for f in os.listdir(settings.UPLOAD_DIR)
                    if not f.startswith(".") and os.path.isfile(os.path.join(settings.UPLOAD_DIR, f))
                ],
                key=lambda f: os.path.getmtime(os.path.join(settings.UPLOAD_DIR, f)),
                reverse=True
            )
            if recent_uploads:
                active_files = [recent_uploads[0]]
                logger.info(f"Scheduled task auto-associated with latest uploaded document: {recent_uploads[0]}")

        async with AsyncSessionLocal() as session:
            # Create a Task record
            task = Task(
                user_id=user_id,
                prompt=f"[SCHEDULED JOB] {clean_prompt}",
                status="EXECUTING"
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)
            task_id = task.id

        # Invoke multi-agent graph
        try:
            initial_state = {
                "task_id": task_id,
                "user_id": user_id,
                "session_id": f"scheduled_{job_id}",
                "user_prompt": clean_prompt,
                "attached_files": active_files,
                "chat_history": [],
                "memory_context": "",
                "execution_plan": {},
                "step_results": {},
                "artifacts": [],
                "retry_count": 0,
                "max_retries": 1,
                "needs_retry": False,
                "retry_instructions": None,
                "events": [],
                "status": "EXECUTING"
            }
            res = await multi_agent_graph.ainvoke(initial_state)

            async with AsyncSessionLocal() as session:
                stmt = select(Task).where(Task.id == task_id)
                db_task = (await session.execute(stmt)).scalar_one_or_none()
                if db_task:
                    db_task.status = "COMPLETED"
                    db_task.final_output = res.get("final_response")
                    db_task.plan = res.get("execution_plan")
                    db_task.completed_at = get_ist_now()
                    await session.commit()

                # Update job last_run
                job_stmt = select(ScheduledJob).where(ScheduledJob.id == job_id)
                db_job = (await session.execute(job_stmt)).scalar_one_or_none()
                if db_job:
                    db_job.last_run = get_ist_now()
                    await session.commit()

            logger.info(f"Scheduled job {job_id} executed successfully.")
        except Exception as e:
            logger.error(f"Error executing scheduled job {job_id}: {e}", exc_info=True)

    def add_job(self, job_id: str, prompt: str, user_id: str, cron_expression: str):
        # cron_expression: e.g. "0 9 * * *"
        fields = cron_expression.strip().split()
        if len(fields) == 5:
            minute, hour, day, month, day_of_week = fields
            trigger = CronTrigger(
                minute=minute,
                hour=hour,
                day=day,
                month=month,
                day_of_week=day_of_week
            )
            self.scheduler.add_job(
                self.execute_scheduled_task,
                trigger=trigger,
                args=[job_id, prompt, user_id],
                id=job_id,
                replace_existing=True
            )
            logger.info(f"Scheduled job added: {job_id} with cron '{cron_expression}'")

    def remove_job(self, job_id: str) -> bool:
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Scheduled job removed from APScheduler: {job_id}")
            return True
        except Exception as e:
            logger.warning(f"Could not remove job {job_id} from APScheduler: {e}")
            return False

    def pause_job(self, job_id: str) -> bool:
        try:
            self.scheduler.pause_job(job_id)
            logger.info(f"Scheduled job paused in APScheduler: {job_id}")
            return True
        except Exception as e:
            logger.warning(f"Could not pause job {job_id} in APScheduler: {e}")
            return False

    def resume_job(self, job_id: str) -> bool:
        try:
            self.scheduler.resume_job(job_id)
            logger.info(f"Scheduled job resumed in APScheduler: {job_id}")
            return True
        except Exception as e:
            logger.warning(f"Could not resume job {job_id} in APScheduler: {e}")
            return False

scheduler_service = TaskSchedulerService()

