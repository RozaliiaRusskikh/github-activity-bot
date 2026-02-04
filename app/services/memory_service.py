from mem0 import Memory
from datetime import datetime
from app.logger import setup_logger

logger = setup_logger(__name__)


class MemoryService:
    def __init__(self):
        """Initialize Supermemory"""
        config = {"version": "v1.1"}
        self.memory = Memory.from_config(config)
        logger.info("Memory service initialized (Supermemory)")

    def add_memory(
        self, user_id: str, question: str, answer: str, commits_analyzed: int
    ):
        """Add Q&A to Supermemory"""
        try:
            logger.info(f"Adding memory for user {user_id}")
            self.memory.add(
                messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                user_id=user_id,
                metadata={
                    "commits_analyzed": commits_analyzed,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            logger.debug(
                f"Memory saved: question={question[:50]}..., commits={commits_analyzed}"
            )
            return True
        except Exception as e:
            logger.error(f"Error saving memory for user {user_id}: {e}")
            return False

    def get_history(self, user_id: str, limit: int = 5) -> list[dict]:
        """Get recent Q&A from Supermemory"""
        try:
            logger.info(f"Retrieving history for user {user_id}, limit={limit}")
            memories = self.memory.get_all(user_id=user_id)
            result = memories[-limit:] if memories else []
            logger.debug(f"Retrieved {len(result)} memories")
            return result
        except Exception as e:
            logger.error(f"Error retrieving memory for user {user_id}: {e}")
            return []

    def health_check(self) -> bool:
        """Check if memory works"""
        try:
            self.memory.add(
                messages=[{"role": "user", "content": "test"}], user_id="health_check"
            )
            logger.debug("Memory health check passed")
            return True
        except Exception as e:
            logger.error(f"Memory health check failed: {e}")
            return False


memory_service = MemoryService()
