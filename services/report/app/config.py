from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Report service configuration."""

    APP_NAME: str = "MediAssist Report Service"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/mediassist"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Storage
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = ""
    MINIO_SECRET_KEY: str = ""
    MINIO_BUCKET: str = "reports"

    # Security
    JWT_SECRET: str = "your-secret-key"
    JWT_ALGORITHM: str = "RS256"

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
