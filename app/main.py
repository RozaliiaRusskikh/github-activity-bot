import discord
import asyncio
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.config import settings
from app.models import Question, Answer, HealthCheck
from app.graph import pipeline_graph, PipelineState
from app.spec_kit import spec_kit_service
from app.services.github_service import github_service
from app.services.ai_service import ai_service
from app.services.memory_service import memory_service
from app.logger import setup_logger

logger = setup_logger(__name__)

# ===== DISCORD BOT =====
intents = discord.Intents.default()
intents.message_content = True
bot = discord.Bot(intents=intents)

# ===== FASTAPI =====


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application lifespan")
    asyncio.create_task(bot.start(settings.discord_token))
    yield
    logger.info("Shutting down application")
    await bot.close()


app = FastAPI(title="GitHub Activity Bot", version="2.0.0", lifespan=lifespan)


@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {
        "message": "GitHub Activity Bot",
        "version": "2.0.0",
        "memory": "mem0 (local)",
    }


@app.get("/health", response_model=HealthCheck)
async def health():
    logger.info("Health check requested")
    health_status = HealthCheck(
        status="healthy",
        discord_bot=bot.is_ready(),
        github_connected=github_service.health_check(),
        ai_connected=ai_service.health_check(),
        memory_connected=memory_service.health_check(),
    )
    logger.debug(f"Health status: {health_status}")
    return health_status


@app.post("/ask", response_model=Answer)
async def ask_api(question: Question):
    logger.info(
        f"API ask request from user {question.user_id}: {question.question[:50]}..."
    )

    initial_state: PipelineState = {
        "question": question.question,
        "user_id": question.user_id,
        "github_data": None,
        "answer": None,
        "commits_analyzed": 0,
        "error": None,
        "force_store": False,
    }

    try:
        config = {"configurable": {"thread_id": question.user_id}}
        result = pipeline_graph.invoke(initial_state, config)

        if result.get("error"):
            logger.error(f"Pipeline error: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])

        logger.info(f"API ask completed successfully for user {question.user_id}")
        return Answer(
            question=question.question,
            answer=result["answer"],
            commits_analyzed=result["commits_analyzed"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ask_api: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 5):
    logger.info(f"History request for user {user_id}, limit={limit}")
    history = memory_service.get_history(user_id, limit)
    return {"user_id": user_id, "history": history}


@app.get("/spec/features")
async def list_features():
    logger.info("Features list requested")
    return spec_kit_service.list_features()


@bot.event
async def on_ready():
    logger.info(f"✅ Discord bot ready: {bot.user} (ID: {bot.user.id})")
    logger.info(f"✅ FastAPI running on http://{settings.api_host}:{settings.api_port}")
    logger.info(f"✅ Memory: mem0 (local storage)")


@bot.event
async def on_application_command(ctx):
    logger.info(
        f"Command /{ctx.command.name} invoked by {ctx.author} (ID: {ctx.author.id})"
    )


@bot.event
async def on_application_command_error(ctx, error):
    logger.error(f"Command error in /{ctx.command.name}: {error}")
    await ctx.respond(f"❌ An error occurred: {str(error)}")


@bot.slash_command(name="ask", description="Ask about GitHub activity")
async def ask_cmd(ctx, question: str):
    await ctx.defer()
    logger.info(f"User {ctx.author} asked: {question[:50]}...")

    try:
        initial_state: PipelineState = {
            "question": question,
            "user_id": str(ctx.author.id),
            "github_data": None,
            "answer": None,
            "commits_analyzed": 0,
            "error": None,
            "force_store": False,
        }

        config = {"configurable": {"thread_id": str(ctx.author.id)}}
        result = pipeline_graph.invoke(initial_state, config)

        if result.get("error"):
            logger.warning(f"Pipeline returned error for user {ctx.author.id}")
            await ctx.respond(f"❌ Error: {result['error']}")
            return

        if result.get("commits_analyzed", 0) == 0:
            logger.info(f"No commits found for user {ctx.author.id}")
            await ctx.respond("❌ No commits in last 24 hours!")
            return

        logger.info(f"Successfully answered question for user {ctx.author.id}")
        await ctx.respond(
            f"📊 **Analyzed {result['commits_analyzed']} commits**\n\n{result['answer']}\n\n"
            f"💾 *Saved to your memory (if important)*"
        )
    except Exception as e:
        logger.error(f"Error in ask command: {e}")
        await ctx.respond(f"❌ Error: {str(e)}")


@bot.slash_command(name="save", description="Force save last answer to memory")
async def save_cmd(ctx):
    """User override: Force store last interaction bypassing filtering"""
    logger.info(f"Save command (user override) from user {ctx.author.id}")

    await ctx.respond(
        "✅ **User Override Enabled**\n\n"
        "Your next interaction will be saved to memory, bypassing content filters.\n"
        "Use this for information you want to remember permanently.\n\n"
        "💡 *Tip: This is a placeholder. In production, track last interaction.*"
    )


@bot.slash_command(name="history", description="Show question history")
async def history_cmd(ctx, limit: int = 5):
    logger.info(f"History command from user {ctx.author.id}, limit={limit}")

    try:
        history = memory_service.get_history(str(ctx.author.id), limit)
        if not history:
            await ctx.respond("📭 No history yet! Ask me about your GitHub activity.")
            return

        text = "\n\n".join(
            [
                f"**Q:** {m.get('messages', [{}])[0].get('content', 'N/A')}\n"
                f"**A:** {m.get('messages', [{}])[1].get('content', 'N/A')[:150]}..."
                for m in history
            ]
        )
        await ctx.respond(f"**Your Memory:**\n\n{text}")
    except Exception as e:
        logger.error(f"Error in history command: {e}")
        await ctx.respond(f"❌ Error: {str(e)}")


@bot.slash_command(name="specify", description="Create feature spec")
async def specify_cmd(ctx, feature_description: str):
    await ctx.defer()
    logger.info(f"Specify command from {ctx.author}: {feature_description[:50]}...")

    try:
        result = spec_kit_service.specify(feature_description)
        await ctx.respond(
            f"✅ **Spec Created**\n"
            f"Feature: {result['feature_name']}\n"
            f"File: `{result['spec_file']}`\n"
            f"Next: `/plan {result['feature_name']}`"
        )
    except Exception as e:
        logger.error(f"Error in specify command: {e}")
        await ctx.respond(f"❌ Error: {str(e)}")


@bot.slash_command(name="plan", description="Create implementation plan")
async def plan_cmd(ctx, feature_name: str):
    await ctx.defer()
    logger.info(f"Plan command for feature: {feature_name}")

    try:
        result = spec_kit_service.plan(feature_name)
        await ctx.respond(
            f"✅ **Plan Created**\n"
            f"Files: `{result['contracts_file']}`, `{result['plan_file']}`\n"
            f"Next: `/task {result['feature_name']}`"
        )
    except Exception as e:
        logger.error(f"Error in plan command: {e}")
        await ctx.respond(f"❌ Error: {str(e)}")


@bot.slash_command(name="task", description="Break into tasks")
async def task_cmd(ctx, feature_name: str):
    await ctx.defer()
    logger.info(f"Task command for feature: {feature_name}")

    try:
        result = spec_kit_service.task(feature_name)
        await ctx.respond(
            f"✅ **Tasks Created**\n"
            f"File: `{result['tasks_file']}`\n"
            f"Ready to implement!"
        )
    except Exception as e:
        logger.error(f"Error in task command: {e}")
        await ctx.respond(f"❌ Error: {str(e)}")


@bot.slash_command(name="features", description="List all features")
async def features_cmd(ctx):
    logger.info(f"Features command from user {ctx.author.id}")

    features = spec_kit_service.list_features()
    if not features:
        await ctx.respond("No features yet!")
        return

    text = "\n".join(
        [
            f"**{f['name']}** - "
            f"Spec: {'✅' if f['has_spec'] else '❌'} | "
            f"Plan: {'✅' if f['has_plan'] else '❌'} | "
            f"Tasks: {'✅' if f['has_tasks'] else '❌'}"
            for f in features
        ]
    )
    await ctx.respond(f"**Features:**\n\n{text}")


@bot.slash_command(name="stats", description="System status")
async def stats_cmd(ctx):
    logger.info(f"Stats command from user {ctx.author.id}")

    github_ok = github_service.health_check()
    ai_ok = ai_service.health_check()
    memory_ok = memory_service.health_check()

    await ctx.respond(
        f"**System Status:**\n"
        f"Discord: ✅\n"
        f"GitHub: {'✅' if github_ok else '❌'}\n"
        f"AI (Gemini): {'✅' if ai_ok else '❌'}\n"
        f"Memory (mem0): {'✅' if memory_ok else '❌'}\n"
        f"LangGraph: ✅\n\n"
        f"**Memory Info:**\n"
        f"• Storage: Local (no API key required)\n"
        f"• Container Tags: `user_{ctx.author.id}`\n"
        f"• Content Filtering: Enabled (LLM)\n\n"
        f"**Stack:** FastAPI, Pydantic, LangChain, LangGraph, Google Gemini, mem0, Spec Kit"
    )


# ===== RUN =====

if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 60)
    logger.info("🚀 Starting GitHub Activity Bot")
    logger.info("=" * 60)
    logger.info("Memory: mem0 (local storage)")
    logger.info("No API key required for memory!")
    logger.info("=" * 60)

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info",
    )
