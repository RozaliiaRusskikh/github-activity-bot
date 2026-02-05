from pydantic import BaseModel, Field
from datetime import datetime


class Commit(BaseModel):
    """Single commit data"""

    repo: str
    message: str
    date: datetime
    sha: str


class GitHubData(BaseModel):
    """Complete GitHub data"""

    commits: list[Commit]
    total_commits: int
    time_range: str


class Question(BaseModel):
    """User question"""

    user_id: str
    question: str
    timestamp: datetime = Field(default_factory=datetime.now)


class Answer(BaseModel):
    """Bot answer"""

    question: str
    answer: str
    commits_analyzed: int
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthCheck(BaseModel):
    """API health check response"""

    status: str
    discord_bot: bool
    github_connected: bool
    ai_connected: bool
    memory_connected: bool = False  # Deprecated, kept for API compatibility
