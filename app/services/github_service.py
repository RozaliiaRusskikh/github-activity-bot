from github import Github
from datetime import datetime, timedelta
from app.models import Commit, GitHubData
from app.config import settings
from app.logger import setup_logger

logger = setup_logger(__name__)


class GitHubService:
    def __init__(self):
        self.client = Github(settings.github_token)
        self.user = self.client.get_user()
        logger.info("GitHub service initialized")

    def get_recent_commits(self, hours: int = 24) -> GitHubData:
        """Fetch commits from last N hours"""
        logger.info(f"Fetching commits from last {hours} hours")
        since = datetime.now() - timedelta(hours=hours)
        commits = []

        # Only recently pushed repos are considered
        for repo in self.user.get_repos(sort='pushed'):
            try:
                repo_commits = repo.get_commits(since=since, author=self.user)
                for commit in list(repo_commits)[:20]:
                    commits.append(
                        Commit(
                            repo=repo.name,
                            message=commit.commit.message,
                            date=commit.commit.author.date,
                            sha=commit.sha[:7],
                        )
                    )
            except Exception as e:
                logger.warning(f"Error fetching from {repo.name}: {e}")
                continue

        logger.info(f"Fetched {len(commits)} commits from {self.user.login}")
        return GitHubData(
            commits=commits,
            total_commits=len(commits),
            time_range=f"last {hours} hours",
        )

    def health_check(self) -> bool:
        """Check if GitHub connection works"""
        try:
            self.user.login
            logger.debug("GitHub health check passed")
            return True
        except Exception as e:
            logger.error(f"GitHub health check failed: {e}")
            return False


github_service = GitHubService()