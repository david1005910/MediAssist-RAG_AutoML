from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    """Analysis service configuration."""

    APP_NAME: str = "MediAssist Analysis Service"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/mediassist"

    # MongoDB
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "mediassist"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # ML Models
    SYMPTOM_MODEL_PATH: str = "./models/symptom_classifier/weights"
    IMAGE_MODEL_PATH: str = "./models/image_analyzer/weights"

    # RAG
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    OPENAI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""

    # Supabase
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # Storage
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = ""
    MINIO_SECRET_KEY: str = ""
    MINIO_BUCKET: str = "reports"

    # Security
    JWT_SECRET: str = "your-secret-key"
    JWT_ALGORITHM: str = "HS256"

    # CORS - 문자열로 저장하고 프로퍼티로 파싱
    CORS_ORIGINS_STR: str = "http://localhost:3000,http://localhost:3003"

    @property
    def CORS_ORIGINS(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.CORS_ORIGINS_STR.split(",") if origin.strip()]

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
