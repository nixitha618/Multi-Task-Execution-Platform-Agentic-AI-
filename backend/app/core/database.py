from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from backend.app.core.config import settings

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

Base = declarative_base()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

from sqlalchemy import text

async def init_db() -> None:
    import backend.app.models.db_models  # noqa: F401 - Register models with Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        def _migrate(connection):
            try:
                res = connection.execute(text("PRAGMA table_info(tasks)"))
                cols = [r[1] for r in res.fetchall()]
                if "session_id" not in cols:
                    connection.execute(text("ALTER TABLE tasks ADD COLUMN session_id VARCHAR(64)"))
            except Exception:
                pass

            try:
                res_users = connection.execute(text("PRAGMA table_info(users)"))
                cols_users = [r[1] for r in res_users.fetchall()]
                if "password_hash" not in cols_users:
                    connection.execute(text("ALTER TABLE users ADD COLUMN password_hash VARCHAR(255)"))
                if "full_name" not in cols_users:
                    connection.execute(text("ALTER TABLE users ADD COLUMN full_name VARCHAR(150)"))
                if "is_active" not in cols_users:
                    connection.execute(text("ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT 1"))
                if "last_login" not in cols_users:
                    connection.execute(text("ALTER TABLE users ADD COLUMN last_login DATETIME"))
                if "updated_at" not in cols_users:
                    connection.execute(text("ALTER TABLE users ADD COLUMN updated_at DATETIME"))
            except Exception:
                pass

        await conn.run_sync(_migrate)
