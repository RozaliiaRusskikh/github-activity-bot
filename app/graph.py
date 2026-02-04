from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.services.github_service import github_service
from app.services.ai_service import ai_service
from app.services.memory_service import memory_service
from app.models import GitHubData
from app.logger import setup_logger

logger = setup_logger(__name__)


class PipelineState(TypedDict):
    """State flowing through LangGraph"""

    question: str
    user_id: str
    github_data: GitHubData | None
    answer: str | None
    commits_analyzed: int
    error: str | None
    force_store: bool  # User override flag


def fetch_github_data(state: PipelineState) -> PipelineState:
    """Node 1: Fetch GitHub commits"""
    logger.info(f"[Pipeline] Fetching GitHub data for user {state['user_id']}")
    try:
        github_data = github_service.get_recent_commits(hours=24)
        logger.info(f"[Pipeline] Fetched {github_data.total_commits} commits")
        return {
            **state,
            "github_data": github_data,
            "commits_analyzed": github_data.total_commits,
        }
    except Exception as e:
        logger.error(f"[Pipeline] GitHub fetch failed: {e}")
        return {**state, "error": f"GitHub fetch failed: {str(e)}"}


def analyze_with_ai(state: PipelineState) -> PipelineState:
    """Node 2: AI analysis with Gemini"""
    logger.info(f"[Pipeline] Analyzing with Gemini: {state['question'][:50]}...")
    try:
        answer = ai_service.answer_question(state["question"], state["github_data"])
        logger.info(f"[Pipeline] Analysis complete, answer length: {len(answer)} chars")
        return {**state, "answer": answer}
    except Exception as e:
        logger.error(f"[Pipeline] AI analysis failed: {e}")
        return {**state, "error": f"AI failed: {str(e)}"}


async def save_to_memory(state: PipelineState) -> PipelineState:
    """Node 3: Save to memory with Supermemory principles"""
    logger.info(f"[Pipeline] Saving to memory for user {state['user_id']}")
    try:
        # Use memory service with proper filtering and container tags
        result = await memory_service.add_memory(
            user_id=state["user_id"],
            question=state["question"],
            answer=state["answer"],
            commits_analyzed=state["commits_analyzed"],
            force_store=state.get("force_store", False),
        )

        if result.get("success"):
            logger.info("[Pipeline] Memory saved successfully")
        elif result.get("filtered"):
            logger.info(f"[Pipeline] Content filtered: {result.get('reason')}")

        return state
    except Exception as e:
        logger.error(f"[Pipeline] Memory save failed: {e}")
        return {**state, "error": f"Memory save failed: {str(e)}"}


def handle_error(state: PipelineState) -> PipelineState:
    """Error handler"""
    logger.error(f"[Pipeline] Error occurred: {state['error']}")
    return state


def should_continue_after_fetch(state: PipelineState) -> Literal["analyze", "error"]:
    """Route after fetch"""
    if state.get("error"):
        logger.warning("[Pipeline] Routing to error handler due to fetch error")
        return "error"
    if state.get("commits_analyzed", 0) == 0:
        logger.warning("[Pipeline] No commits found, routing to error")
        return "error"
    logger.info("[Pipeline] Routing to AI analysis")
    return "analyze"


def create_pipeline_graph():
    """Build LangGraph workflow"""
    logger.info("Building LangGraph pipeline")
    workflow = StateGraph(PipelineState)

    # Add nodes
    workflow.add_node("fetch", fetch_github_data)
    workflow.add_node("analyze", analyze_with_ai)
    workflow.add_node("save", save_to_memory)
    workflow.add_node("error", handle_error)

    # Set entry
    workflow.set_entry_point("fetch")

    # Add edges
    workflow.add_conditional_edges(
        "fetch", should_continue_after_fetch, {"analyze": "analyze", "error": "error"}
    )
    workflow.add_edge("analyze", "save")
    workflow.add_edge("save", END)
    workflow.add_edge("error", END)

    # Compile
    memory = MemorySaver()
    logger.info("LangGraph pipeline compiled successfully")
    return workflow.compile(checkpointer=memory)


pipeline_graph = create_pipeline_graph()
