from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.models import GitHubData
from app.config import settings
from app.logger import setup_logger

logger = setup_logger(__name__)


class AIService:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro", google_api_key=settings.google_api_key, temperature=0.7
        )
        logger.info("🤖 AI Service initialized with Google Gemini Pro")

    def answer_question(self, question: str, github_data: GitHubData) -> str:
        """Use LangChain with Gemini to answer based on GitHub data"""

        logger.info(f"Processing question: {question[:50]}...")

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
            response = self.llm.invoke(messages)
            logger.info("Successfully received response from Gemini")
            return response.content
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
