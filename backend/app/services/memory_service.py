import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import select, delete, desc
from backend.app.core.database import AsyncSessionLocal
from backend.app.models.db_models import UserMemory, User, ConversationMessage

logger = logging.getLogger(__name__)

class MemoryService:
    """Manages short-term and long-term user memory in SQLite."""

    async def get_or_create_user(self, user_id: str, username: Optional[str] = None) -> User:
        uname = username or f"usr_{user_id}"
        async with AsyncSessionLocal() as session:
            stmt = select(User).where((User.id == user_id) | (User.username == uname))
            res = await session.execute(stmt)
            user = res.scalar_one_or_none()
            if not user:
                user = User(
                    id=user_id,
                    username=uname,
                    preferences={}
                )
                session.add(user)
                await session.commit()
                await session.refresh(user)
            return user

    async def add_memory(
        self,
        user_id: str,
        category: str,
        key: str,
        value: str,
        importance: float = 1.0
    ) -> UserMemory:
        await self.get_or_create_user(user_id)
        async with AsyncSessionLocal() as session:
            # Check if key already exists for this user (prevent duplicate keys across categories)
            stmt = select(UserMemory).where(
                UserMemory.user_id == user_id,
                UserMemory.key == key
            )
            res = await session.execute(stmt)
            existing = res.scalar_one_or_none()

            if existing:
                existing.value = value
                existing.category = category
                existing.importance = importance
                await session.commit()
                await session.refresh(existing)
                return existing
            else:
                mem = UserMemory(
                    user_id=user_id,
                    category=category,
                    key=key,
                    value=value,
                    importance=importance
                )
                session.add(mem)
                await session.commit()
                await session.refresh(mem)
                return mem

    async def get_memories(
        self,
        user_id: str,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            query = select(UserMemory).where(UserMemory.user_id == user_id)
            if category:
                query = query.where(UserMemory.category == category)
            query = query.order_by(desc(UserMemory.importance), desc(UserMemory.updated_at)).limit(limit)

            res = await session.execute(query)
            memories = res.scalars().all()
            return [
                {
                    "id": m.id,
                    "user_id": m.user_id,
                    "category": m.category,
                    "key": m.key,
                    "value": m.value,
                    "importance": m.importance,
                    "created_at": m.created_at,
                    "updated_at": m.updated_at
                }
                for m in memories
            ]

    async def get_user_preferences(self, user_id: str) -> Dict[str, str]:
        """
        Returns all stored preferences, facts, and custom rules for a user as a key -> value dictionary.
        Supports any type of preference added by the user.
        """
        async with AsyncSessionLocal() as session:
            stmt = select(UserMemory).where(
                UserMemory.user_id == user_id,
                UserMemory.category.in_(["preference", "fact", "rule"])
            ).order_by(desc(UserMemory.importance), desc(UserMemory.updated_at))
            res = await session.execute(stmt)
            memories = res.scalars().all()
            return {m.key: m.value for m in memories}

    async def get_user_preference(self, user_id: str, key_pattern: str) -> Optional[str]:
        """
        Flexible lookup for a user preference or fact by key or substring (case-insensitive).
        For instance, matching 'summary_style', 'college', 'email_signature', 'chart_theme', 'tone'.
        """
        prefs = await self.get_user_preferences(user_id)
        if not prefs:
            return None

        # 1. Exact match (case-insensitive)
        for k, v in prefs.items():
            if k.lower() == key_pattern.lower():
                return v

        # 2. Substring match (either direction)
        clean_pat = key_pattern.lower().replace("_", " ").strip()
        for k, v in prefs.items():
            clean_k = k.lower().replace("_", " ").strip()
            if clean_pat in clean_k or clean_k in clean_pat:
                return v

        return None

    async def get_memory_context_string(self, user_id: str) -> str:
        """Formats all user memories, preferences, and facts into an organized context snippet."""
        async with AsyncSessionLocal() as session:
            stmt_prefs = select(UserMemory).where(
                UserMemory.user_id == user_id,
                UserMemory.category.in_(["preference", "fact", "rule"])
            ).order_by(desc(UserMemory.importance), desc(UserMemory.updated_at))
            res_prefs = await session.execute(stmt_prefs)
            user_settings = res_prefs.scalars().all()

            stmt_hist = select(UserMemory).where(
                UserMemory.user_id == user_id,
                UserMemory.category == "task_history"
            ).order_by(desc(UserMemory.updated_at)).limit(5)
            res_hist = await session.execute(stmt_hist)
            history = res_hist.scalars().all()

        if not user_settings and not history:
            return "No previous memory stored for this user."

        sections = []
        if user_settings:
            styling = []
            facts = []
            rules = []
            for m in user_settings:
                k_lower = m.key.lower()
                if any(w in k_lower for w in ["style", "theme", "format", "tone", "signature", "length", "bullet", "language", "takeaway"]):
                    styling.append(f"- [PREFERENCE] {m.key}: {m.value}")
                elif m.category == "rule" or "rule" in k_lower:
                    rules.append(f"- [RULE] {m.key}: {m.value}")
                else:
                    facts.append(f"- [FACT] {m.key}: {m.value}")

            if styling:
                sections.append("### Active Output & Styling Preferences (MANDATORY TO FOLLOW):\n" + "\n".join(styling))
            if facts:
                sections.append("### User Profile & Identity Facts (Background Information):\n" + "\n".join(facts))
            if rules:
                sections.append("### Operational Rules & Directives:\n" + "\n".join(rules))

        if history:
            hist_lines = [f"- Task: {h.key} | {h.value[:140]}..." for h in history]
            sections.append("### Recent Workflow History:\n" + "\n".join(hist_lines))

        return "\n\n".join(sections)

    async def delete_memory(self, memory_id: str) -> bool:
        async with AsyncSessionLocal() as session:
            stmt = delete(UserMemory).where(UserMemory.id == memory_id)
            res = await session.execute(stmt)
            await session.commit()
            return res.rowcount > 0

    # ==========================================
    # Persistent Conversation History Management
    # ==========================================

    async def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Persists a user or assistant message to SQLite conversation_messages table."""
        async with AsyncSessionLocal() as session:
            msg = ConversationMessage(
                session_id=session_id,
                role=role,
                content=content,
                message_metadata=metadata or {}
            )
            session.add(msg)
            await session.commit()
            await session.refresh(msg)
            return {
                "id": msg.id,
                "session_id": msg.session_id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "metadata": msg.message_metadata
            }

    async def get_session_messages(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves chronological messages for the given session_id."""
        async with AsyncSessionLocal() as session:
            query = select(ConversationMessage).where(
                ConversationMessage.session_id == session_id
            ).order_by(ConversationMessage.timestamp.asc(), ConversationMessage.id.asc())
            if limit:
                query = query.limit(limit)
            res = await session.execute(query)
            messages = res.scalars().all()
            return [
                {
                    "id": m.id,
                    "session_id": m.session_id,
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "metadata": m.message_metadata
                }
                for m in messages
            ]

    async def get_previous_user_question(
        self,
        session_id: str,
        current_message_id: Optional[int] = None,
        current_prompt: Optional[str] = None,
        offset: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves the prior user question in the session.
        offset=1: immediately preceding user question.
        offset=2: two questions ago.
        Excludes current_message_id and any message matching current_prompt.
        """
        async with AsyncSessionLocal() as session:
            query = select(ConversationMessage).where(
                ConversationMessage.session_id == session_id,
                ConversationMessage.role == "user"
            )
            if current_message_id is not None:
                query = query.where(ConversationMessage.id != current_message_id)
            if current_prompt:
                query = query.where(ConversationMessage.content != current_prompt)

            query = query.order_by(desc(ConversationMessage.timestamp), desc(ConversationMessage.id))
            res = await session.execute(query)
            user_messages = res.scalars().all()

            if not user_messages:
                return None

            target_idx = offset - 1
            if 0 <= target_idx < len(user_messages):
                msg = user_messages[target_idx]
                return {
                    "id": msg.id,
                    "session_id": msg.session_id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "metadata": msg.message_metadata
                }
            return None

    async def get_first_user_question(
        self,
        session_id: str,
        current_message_id: Optional[int] = None,
        current_prompt: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieves the very first user question of the session."""
        async with AsyncSessionLocal() as session:
            query = select(ConversationMessage).where(
                ConversationMessage.session_id == session_id,
                ConversationMessage.role == "user"
            )
            if current_message_id is not None:
                query = query.where(ConversationMessage.id != current_message_id)
            if current_prompt:
                query = query.where(ConversationMessage.content != current_prompt)

            query = query.order_by(ConversationMessage.timestamp.asc(), ConversationMessage.id.asc())
            res = await session.execute(query)
            msg = res.scalars().first()
            if msg:
                return {
                    "id": msg.id,
                    "session_id": msg.session_id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "metadata": msg.message_metadata
                }
            return None

    async def get_all_user_questions(
        self,
        session_id: str,
        current_message_id: Optional[int] = None,
        current_prompt: Optional[str] = None
    ) -> List[str]:
        """Retrieves list of all user questions in the session (excluding current)."""
        async with AsyncSessionLocal() as session:
            query = select(ConversationMessage).where(
                ConversationMessage.session_id == session_id,
                ConversationMessage.role == "user"
            )
            if current_message_id is not None:
                query = query.where(ConversationMessage.id != current_message_id)
            if current_prompt:
                query = query.where(ConversationMessage.content != current_prompt)

            query = query.order_by(ConversationMessage.timestamp.asc(), ConversationMessage.id.asc())
            res = await session.execute(query)
            return [m.content for m in res.scalars().all()]

    async def get_previous_assistant_answer(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieves the immediately preceding assistant response."""
        async with AsyncSessionLocal() as session:
            query = select(ConversationMessage).where(
                ConversationMessage.session_id == session_id,
                ConversationMessage.role == "assistant"
            ).order_by(desc(ConversationMessage.timestamp), desc(ConversationMessage.id))
            res = await session.execute(query)
            msg = res.scalars().first()
            if msg:
                return {
                    "id": msg.id,
                    "session_id": msg.session_id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "metadata": msg.message_metadata
                }
            return None

    async def get_conversation_context(
        self,
        session_id: str,
        window: int = 10,
        exclude_id: Optional[int] = None
    ) -> str:
        """Formats the most recent conversation messages into a clean LLM context snippet."""
        async with AsyncSessionLocal() as session:
            query = select(ConversationMessage).where(
                ConversationMessage.session_id == session_id
            )
            if exclude_id is not None:
                query = query.where(ConversationMessage.id != exclude_id)

            query = query.order_by(desc(ConversationMessage.timestamp), desc(ConversationMessage.id)).limit(window)
            res = await session.execute(query)
            messages = list(res.scalars().all())
            messages.reverse()

            if not messages:
                return ""

            lines = ["### Conversation History (Current Session):"]
            for m in messages:
                speaker = "User" if m.role == "user" else "Assistant"
                content_preview = (m.content or "").strip()
                if len(content_preview) > 600:
                    content_preview = content_preview[:600] + " ... [truncated]"
                lines.append(f"{speaker}: {content_preview}")

            full_context = "\n".join(lines)
            if len(full_context) > 2500:
                full_context = lines[0] + "\n... [earlier conversation omitted] ...\n" + full_context[-2200:]
            return full_context

    async def clear_session(self, session_id: str) -> int:
        """Deletes all messages for a given session."""
        async with AsyncSessionLocal() as session:
            stmt = delete(ConversationMessage).where(ConversationMessage.session_id == session_id)
            res = await session.execute(stmt)
            await session.commit()
            return res.rowcount

memory_service = MemoryService()

