import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.models import GitHubData
from app.config import settings
from app.logger import setup_logger

logger = setup_logger(__name__)


class AIService:
    def __init__(self):
        # Use gemini-pro which is the stable, widely available model
        # Alternative models: gemini-1.5-pro, gemini-1.5-flash (if available)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", google_api_key=settings.google_api_key, temperature=0.7, version="v1"
        )
        logger.info("🤖 AI Service initialized with Google Gemini Pro")

    async def answer_question(self, question: str, github_data: GitHubData) -> str:
        """Use LangChain with Gemini to answer based on GitHub data"""

        logger.info(f"Processing question: {question[:50]}...")

        if not github_data or not github_data.commits:
            logger.warning("No GitHub data or commits provided")
            return "No recent commits found."

        context = "\n".join(
            [
                f"[{commit.repo}] {commit.message} "
                f"({commit.date.strftime('%I:%M %p')})"
                for commit in github_data.commits
            ]
        )

        if not context:
            logger.warning("No commits found in context")
            return "No recent commits found."

        messages = [
            SystemMessage(
                content=(
                    "You are a helpful assistant answering questions about "
                    "a developer's GitHub activity. Be concise."
                )
            ),
            HumanMessage(
                content=(
                    f"Based on these commits:\n\n{context}\n\n"
                    f"Answer: {question}\n\nKeep it concise."
                )
            ),
        ]

        try:
            logger.debug("Sending request to Gemini API")
            # Run blocking LLM call in thread pool to avoid blocking event loop
            # Add timeout for AI API calls (30 seconds)
            response = await asyncio.wait_for(
                asyncio.to_thread(self.llm.invoke, messages),
                timeout=30.0
            )
            logger.info("Successfully received response from Gemini")
            return response.content
        except asyncio.TimeoutError:
            logger.error("Gemini API call timed out")
            raise Exception("AI service request timed out. Please try again.")
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            raise

    def health_check(self) -> bool:
        """Check if Gemini service works"""
        try:
            self.llm.invoke([HumanMessage(content="test")])
            logger.debug("AI health check passed")
            return True
        except Exception as e:
            logger.error(f"AI health check failed: {e}")
            return False


ai_service = AIService()
