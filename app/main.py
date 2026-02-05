import discord
import asyncio
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.config import settings
from app.models import Question, Answer, HealthCheck
from app.graph import pipeline_graph, PipelineState
from app.services.github_service import github_service
from app.services.ai_service import ai_service
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
    }


@app.get("/health", response_model=HealthCheck)
async def health():
    logger.info("Health check requested")
    health_status = HealthCheck(
        status="healthy",
        discord_bot=bot.is_ready(),
        github_connected=github_service.health_check(),
        ai_connected=ai_service.health_check(),
        memory_connected=False,
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
    }

    try:
        config = {"configurable": {"thread_id": question.user_id}}
        result = await pipeline_graph.ainvoke(initial_state, config)

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


@bot.event
async def on_ready():
    logger.info(f"✅ Discord bot ready: {bot.user} (ID: {bot.user.id})")
    logger.info(f"✅ FastAPI running on http://{settings.api_host}:{settings.api_port}")


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
        }

        config = {"configurable": {"thread_id": str(ctx.author.id)}}
        
        # Add timeout to prevent indefinite hanging (60 seconds)
        try:
            result = await asyncio.wait_for(
                pipeline_graph.ainvoke(initial_state, config),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            logger.error(f"Pipeline timeout for user {ctx.author.id}")
            await ctx.respond("❌ Request timed out. Please try again.")
            return

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
            f"📊 **Analyzed {result['commits_analyzed']} commits**\n\n{result['answer']}"
        )
    except asyncio.TimeoutError:
        # Already handled above, but just in case
        logger.error(f"Timeout in ask command for user {ctx.author.id}")
        await ctx.respond("❌ Request timed out. Please try again.")
    except Exception as e:
        logger.error(f"Error in ask command: {e}", exc_info=True)
        await ctx.respond(f"❌ Error: {str(e)}")


# ===== RUN =====

if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 60)
    logger.info("🚀 Starting GitHub Activity Bot")
    logger.info("=" * 60)

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info",
    )
