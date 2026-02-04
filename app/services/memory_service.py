from mem0 import Memory
from typing import List
from datetime import datetime
from app.logger import setup_logger
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from app.config import settings
import json

logger = setup_logger(__name__)


class MemoryService:
    """
    Memory service following Supermemory organizational standards:
    - Container-based isolation (user_{id})
    - Required metadata schema
    - Content filtering with user override
    - Access control at application layer

    Uses mem0 (local) - no API key required
    """

    def __init__(self):
        """Initialize mem0 with local storage"""
        config = {"version": "v1.1"}
        self.memory = Memory.from_config(config)

        # Initialize LLM for content filtering (Layer 2)
        self.filter_llm = ChatGoogleGenerativeAI(
            model="gemini-pro", google_api_key=settings.google_api_key, temperature=0.7
        )

        logger.info("Memory service initialized (mem0 with Supermemory principles)")

    def _build_container_tags(self, user_id: str) -> List[str]:
        """
        Build container tags following Supermemory naming convention:
        - user_{userId} for private
        """
        return [f"user_{user_id}"]

    def _build_metadata(self, user_id: str, commits_analyzed: int) -> dict:
        """Build required metadata schema (Supermemory principles)"""
        return {
            "type": "conversation",
            "source": "discord",
            "created_by": user_id,
            "created_at": datetime.now().isoformat(),
            "sensitivity": "private",
            "commits_analyzed": commits_analyzed,
            "tags": ["github", "activity", "bot-interaction"],
        }

    async def _should_store_content(self, content: str, source: str) -> dict:
        """
        Layer 2: Application general content filter
        Evaluates if content is important enough to store
        """
        try:
            logger.debug(f"Evaluating content from {source}: {content[:50]}...")

            prompt = f"""Evaluate if this {source} content is important enough to store permanently.

Content:
{content}

Respond with JSON: {{"store": true/false, "reason": "brief explanation"}}

STORE if it contains:
- Decisions, action items, or commitments
- Important information for future reference
- User preferences or settings
- Questions and answers with learning value
- Key context that would be valuable later

DO NOT STORE if it's:
- Simple acknowledgments ("ok", "thanks", "got it")
- Already captured information
- Temporary or transient data
- Out-of-context fragments"""

            response = self.filter_llm.invoke([HumanMessage(content=prompt)])

            # Parse JSON response
            result = json.loads(response.content.strip())

            logger.info(f"Filter decision: {result['store']} - {result['reason']}")
            return result

        except Exception as e:
            logger.error(f"Error in content filtering: {e}")
            # Default to storing on filter error
            return {"store": True, "reason": "Filter error - storing by default"}

    async def add_memory(
        self,
        user_id: str,
        question: str,
        answer: str,
        commits_analyzed: int,
        force_store: bool = False,
    ) -> dict:
        """
        Add Q&A to memory with Supermemory principles

        Args:
            user_id: User identifier
            question: User's question
            answer: Bot's answer
            commits_analyzed: Number of commits analyzed
            force_store: User override - bypass filtering

        Returns:
            dict with success status and filter decision
        """
        try:
            # Build container tags following naming convention
            container_tags = self._build_container_tags(user_id)

            logger.info(
                f"Adding memory for user {user_id}, containers: {container_tags}"
            )

            # Content to evaluate
            content = f"Question: {question}\nAnswer: {answer}"

            # User Override: Skip filtering if explicitly marked important
            if force_store:
                logger.info("User override: Storing without filtering")
                filter_decision = {"store": True, "reason": "User override"}
                should_store = True
            else:
                # Layer 2: Application-layer content filter
                filter_decision = await self._should_store_content(content, "discord")
                should_store = filter_decision.get("store", False)

            if not should_store:
                logger.info("Content filtered out - not storing")
                return {
                    "success": False,
                    "filtered": True,
                    "reason": filter_decision.get("reason"),
                }

            # Build metadata following required schema
            metadata = self._build_metadata(user_id, commits_analyzed)

            # Store in mem0
            self.memory.add(
                messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                user_id=user_id,
                metadata=metadata,
            )

            logger.debug(
                f"Memory saved: question={question[:50]}..., commits={commits_analyzed}"
            )
            return {
                "success": True,
                "filtered": False,
                "reason": filter_decision.get("reason"),
            }

        except Exception as e:
            logger.error(f"Error saving memory for user {user_id}: {e}")
            return {"success": False, "error": str(e)}

    def get_history(self, user_id: str, limit: int = 5) -> List[dict]:
        """
        Get recent Q&A from memory with proper scoping

        Args:
            user_id: User identifier
            limit: Maximum number of memories to retrieve
        """
        try:
            # Build container tags for scoped access
            container_tags = self._build_container_tags(user_id)

            logger.info(
                f"Retrieving history for user {user_id}, containers: {container_tags}, limit={limit}"
            )

            # Get all memories for this user
            memories = self.memory.get_all(user_id=user_id)

            # Return last N memories
            result = memories[-limit:] if memories else []

            logger.debug(f"Retrieved {len(result)} memories")
            return result

        except Exception as e:
            logger.error(f"Error retrieving memory for user {user_id}: {e}")
            return []

    def search_memories(self, user_id: str, query: str, limit: int = 10) -> List[dict]:
        """
        Search memories with proper scoping

        Args:
            user_id: User identifier
            query: Search query
            limit: Maximum results
        """
        try:
            logger.info(
                f"Searching memories for user {user_id}, query: {query[:50]}..."
            )

            # Search with user scope
            results = self.memory.search(query=query, user_id=user_id, limit=limit)

            logger.info(f"Found {len(results) if results else 0} relevant memories")
            return results if results else []

        except Exception as e:
            logger.error(f"Error searching memories for user {user_id}: {e}")
            return []

    def health_check(self) -> bool:
        """Check if memory service works"""
        try:
            # Simple health check
            test_metadata = {
                "type": "note",
                "source": "system",
                "created_by": "health_check",
                "created_at": datetime.now().isoformat(),
                "sensitivity": "private",
            }

            self.memory.add(
                messages=[{"role": "user", "content": "health_check"}],
                user_id="health_check",
                metadata=test_metadata,
            )
            logger.debug("Memory health check passed")
            return True
        except Exception as e:
            logger.error(f"Memory health check failed: {e}")
            return False


memory_service = MemoryService()
