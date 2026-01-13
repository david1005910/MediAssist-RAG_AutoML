from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Analysis service configuration."""

    APP_NAME: str = "MediAssist Analysis Service"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/mediassist"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # ML Models
    SYMPTOM_MODEL_PATH: str = "./models/symptom_classifier/weights"
    IMAGE_MODEL_PATH: str = "./models/image_analyzer/weights"

    # RAG
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    OPENAI_API_KEY: str = ""

    # Supabase
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # Security
    JWT_SECRET: str = "your-secret-key"
    JWT_ALGORITHM: str = "RS256"

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:3003", "http://localhost:3004", "http://localhost:3005"]

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
