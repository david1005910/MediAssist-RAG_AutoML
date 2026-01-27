from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Auth service configuration."""

    APP_NAME: str = "MediAssist Auth Service"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/mediassist"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Security
    JWT_SECRET: str = "your-secret-key"
    JWT_ALGORITHM: str = "RS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
