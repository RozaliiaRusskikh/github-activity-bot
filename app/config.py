from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings loaded from .env"""
    
    # Discord
    discord_token: str
    
    # GitHub
    github_token: str
    
    # Google Gemini
    google_api_key: str
    
    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Singleton instance
settings = Settings()