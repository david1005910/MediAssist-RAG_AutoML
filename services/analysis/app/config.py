from pydantic_settings import BaseSettings
from pydantic import field_validator
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
    JWT_ALGORITHM: str = "HS256"

    # CORS - 환경변수에서 쉼표로 구분된 문자열을 리스트로 파싱
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:3003", "http://localhost:3004", "http://localhost:3005"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
