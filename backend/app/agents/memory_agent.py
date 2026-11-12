import re
from typing import Dict, Any, Optional, List
from backend.app.agents.base import BaseAgent
from backend.app.services.memory_service import memory_service

class MemoryAgent(BaseAgent):
    """Agent 5: Reads user memory/preferences before execution and persists new knowledge post-execution."""

    def __init__(self):
        super().__init__(
            name="MemoryAgent",
            description="Maintains short-term and long-term user context, preferences, and conversation history."
        )

    async def retrieve_context(self, user_id: str) -> str:
        return await memory_service.get_memory_context_string(user_id)

    async def retrieve_conversation_context(self, session_id: str, window: int = 10, exclude_id: Optional[int] = None) -> str:
        if not session_id:
            return ""
        return await memory_service.get_conversation_context(session_id, window=window, exclude_id=exclude_id)

    async def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        if not session_id:
            return []
        return await memory_service.get_session_messages(session_id)

    async def persist_conversation_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return await memory_service.save_message(session_id, role, content, metadata)

    async def persist_execution_memory(self, user_id: str, prompt: str, outcome: str, session_id: Optional[str] = None):
        # 1. Store high-level task memory in UserMemory
        key = f"Task_{prompt[:30].strip().replace(' ', '_')}"
        val = f"Prompt: {prompt} | Outcome: {outcome[:120]}..."
        await memory_service.add_memory(
            user_id=user_id,
            category="task_history",
            key=key,
            value=val,
            importance=0.8
        )
        # 2. Store assistant response in conversation_messages if session_id is active
        if session_id:
            await memory_service.save_message(
                session_id=session_id,
                role="assistant",
                content=outcome,
                metadata={"user_id": user_id, "prompt_preview": prompt[:60]}
            )

    async def resolve_deterministic_memory_query(
        self,
        session_id: str,
        prompt: str,
        current_message_id: Optional[int] = None,
        user_id: str = "default_user"
    ) -> Optional[Dict[str, Any]]:
        """
        Deterministically resolves conversational memory questions
        (e.g., 'What was my previous question?', 'What is my college?',
        'What is my summary style?', 'List my preferences') directly from SQLite.
        """
        clean_p = prompt.strip().lower().rstrip("?.! ")

        all_user_prefs = await memory_service.get_user_preferences(user_id)

        # A. Check for "what preferences do I have", "list my preferences", "show my memory"
        if any(pat in clean_p for pat in [
            "what preferences", "list my preferences", "show my preferences",
            "what is in my memory", "what memories do i have", "my stored preferences",
            "show stored memory", "list memories", "what do you know about me",
            "what are my preferences", "what is my memory", "my preferences"
        ]):
            if all_user_prefs:
                lines = [f"- **{k.replace('_', ' ').title()}:** {v}" for k, v in all_user_prefs.items()]
                return {
                    "query_type": "all_preferences",
                    "found": True,
                    "target_content": all_user_prefs,
                    "formatted_answer": "### Your Active Preferences & Stored Profile Settings:\n\n" + "\n".join(lines)
                }
            else:
                return {
                    "query_type": "all_preferences",
                    "found": False,
                    "target_content": {},
                    "formatted_answer": "You do not have any preferences or profile settings stored in your memory yet."
                }

        # B. Check for queries asking for specific stored attributes (e.g. "what is my college", "what is my signature", "what is my summary style")
        if all_user_prefs and not any(q_word in clean_p for q_word in ["previous", "last", "first", "ago", "repeat", "how to"]):
            # 1. Match stored keys directly against prompt words/phrases
            sorted_keys = sorted(all_user_prefs.keys(), key=lambda x: len(x), reverse=True)
            for k in sorted_keys:
                val = all_user_prefs[k]
                k_clean = k.lower().replace("_", " ").strip()
                # Check if k_clean or primary word is mentioned in query along with query triggers
                if (k_clean in clean_p or (len(k_clean) > 3 and k_clean in clean_p.split())) and any(trig in clean_p for trig in ["what", "which", "where", "tell", "show", "who", "my", "is"]):
                    return {
                        "query_type": "user_preference_lookup",
                        "found": True,
                        "target_content": val,
                        "formatted_answer": f"According to your stored profile and preferences, your **{k.replace('_', ' ').title()}** is:\n\n> **{val}**"
                    }

        # 2. Pattern matching for generic "what is my <attribute>"
        pref_pattern = re.search(r"\b(?:what is|what's|which|tell me|show me)\s+(?:my|the|about my)?\s+([a-zA-Z0-9_\- ]+)\b", clean_p)
        if pref_pattern:
            candidate_key = pref_pattern.group(1).strip()
            # Ignore conversational queries like "my previous question"
            if not any(q_word in candidate_key for q_word in ["previous", "last", "first", "ago", "question", "answer", "query", "step"]):
                found_val = await memory_service.get_user_preference(user_id, candidate_key)
                if found_val:
                    return {
                        "query_type": "user_preference_lookup",
                        "found": True,
                        "target_content": found_val,
                        "formatted_answer": f"According to your stored profile and preferences, your **{candidate_key.title()}** is:\n\n> **{found_val}**"
                    }

        if not session_id:
            return None

        # 1. Check for relative N-questions ago: "two questions ago" / "2 questions ago" / "3 questions ago"
        n_match = re.search(r"\b(\d+|two|three|four|five)\s+questions?\s+ago\b", clean_p)
        if n_match:
            word_map = {"two": 2, "three": 3, "four": 4, "five": 5}
            raw_n = n_match.group(1)
            offset = word_map.get(raw_n, int(raw_n) if raw_n.isdigit() else 2)
            prev_msg = await memory_service.get_previous_user_question(
                session_id=session_id,
                current_message_id=current_message_id,
                current_prompt=prompt,
                offset=offset
            )
            if prev_msg:
                return {
                    "query_type": "n_questions_ago",
                    "offset": offset,
                    "found": True,
                    "target_content": prev_msg["content"],
                    "formatted_answer": f"The question you asked {offset} questions ago was:\n\n> \"{prev_msg['content']}\""
                }
            else:
                return {
                    "query_type": "n_questions_ago",
                    "offset": offset,
                    "found": False,
                    "target_content": None,
                    "formatted_answer": f"I don't have a question from {offset} questions ago in this conversation session."
                }

        # 2. Check for "first question"
        if any(pat in clean_p for pat in [
            "first question", "first query", "what was the first question", "what was my first question", "first question i asked"
        ]):
            first_msg = await memory_service.get_first_user_question(
                session_id=session_id,
                current_message_id=current_message_id,
                current_prompt=prompt
            )
            if first_msg:
                return {
                    "query_type": "first_question",
                    "found": True,
                    "target_content": first_msg["content"],
                    "formatted_answer": f"Your first question in this conversation was:\n\n> \"{first_msg['content']}\""
                }
            else:
                return {
                    "query_type": "first_question",
                    "found": False,
                    "target_content": None,
                    "formatted_answer": "There are no previous questions recorded in the current conversation history."
                }

        # 3. Check for "previous question" / "last question" / "what did I ask earlier" / "question before this"
        if any(pat in clean_p for pat in [
            "previous question", "previous query", "last question", "what was my previous question",
            "what is my previous question", "what did i ask earlier", "what did i ask you earlier",
            "what was the question before this", "what did i ask before", "what was my last question",
            "what did i ask last", "repeat my previous question"
        ]):
            prev_msg = await memory_service.get_previous_user_question(
                session_id=session_id,
                current_message_id=current_message_id,
                current_prompt=prompt,
                offset=1
            )
            if prev_msg:
                return {
                    "query_type": "previous_question",
                    "found": True,
                    "target_content": prev_msg["content"],
                    "formatted_answer": f"Your previous question was:\n\n> \"{prev_msg['content']}\""
                }
            else:
                return {
                    "query_type": "previous_question",
                    "found": False,
                    "target_content": None,
                    "formatted_answer": "I don't have any prior question from you in the current conversation history to retrieve."
                }

        # 4. Check for all questions: "what questions have I asked" / "list of questions"
        if any(pat in clean_p for pat in [
            "what questions have i asked", "what questions did i ask", "list all questions i asked",
            "all questions i asked", "what have i asked so far"
        ]):
            all_q = await memory_service.get_all_user_questions(
                session_id=session_id,
                current_message_id=current_message_id,
                current_prompt=prompt
            )
            if all_q:
                bullets = "\n".join(f"{i+1}. \"{q}\"" for i, q in enumerate(all_q))
                return {
                    "query_type": "all_questions",
                    "found": True,
                    "target_content": all_q,
                    "formatted_answer": f"Here are the questions you have asked in this conversation:\n\n{bullets}"
                }
            else:
                return {
                    "query_type": "all_questions",
                    "found": False,
                    "target_content": [],
                    "formatted_answer": "You haven't asked any previous questions in this conversation session yet."
                }

        # 5. Check for "previous answer" / "what did you tell me earlier"
        if any(pat in clean_p for pat in [
            "previous answer", "last answer", "what was your previous answer", "what was your last answer"
        ]):
            prev_ans = await memory_service.get_previous_assistant_answer(session_id=session_id)
            if prev_ans:
                return {
                    "query_type": "previous_answer",
                    "found": True,
                    "target_content": prev_ans["content"],
                    "formatted_answer": f"My previous answer was:\n\n{prev_ans['content']}"
                }
            else:
                return {
                    "query_type": "previous_answer",
                    "found": False,
                    "target_content": None,
                    "formatted_answer": "I haven't provided any previous answers in this conversation session yet."
                }

        return None

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_id = state.get("user_id", "default_user")
        session_id = state.get("session_id")
        action = state.get("memory_action", "retrieve")

        if action == "retrieve":
            ctx = await self.retrieve_context(user_id)
            conv_ctx = await self.retrieve_conversation_context(session_id) if session_id else ""
            return {
                "memory_context": ctx,
                "conversation_context": conv_ctx
            }
        elif action == "persist":
            prompt = state.get("user_prompt", "")
            final_resp = state.get("final_response", "")
            await self.persist_execution_memory(user_id, prompt, final_resp, session_id=session_id)
            return {"memory_persisted": True}

        return {}

memory_agent = MemoryAgent()

