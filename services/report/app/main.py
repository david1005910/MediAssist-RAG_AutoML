import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Service health state
service_state: Dict[str, Any] = {
    "database": False,
    "storage": False,
    "startup_complete": False,
}


async def check_database_connection() -> bool:
    """Check PostgreSQL connection status."""
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(settings.DATABASE_URL, echo=False)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        await engine.dispose()
        return True
    except Exception as e:
        logger.warning(f"Database connection check failed: {e}")
        return False


async def check_storage_connection() -> bool:
    """Check MinIO/S3 storage connection status."""
    try:
        from minio import Minio

        if not settings.MINIO_ENDPOINT or settings.MINIO_ENDPOINT == "localhost:9000":
            return False

        client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=True,
        )
        client.list_buckets()
        return True
    except Exception as e:
        logger.warning(f"Storage connection check failed: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    logger.info("=" * 50)
    logger.info("MediAssist Report Service Starting...")
    logger.info("=" * 50)

    service_state["database"] = await check_database_connection()
    logger.info(f"  PostgreSQL: {'Connected' if service_state['database'] else 'Not available'}")

    service_state["storage"] = await check_storage_connection()
    logger.info(f"  Object Storage: {'Connected' if service_state['storage'] else 'Not available'}")

    service_state["startup_complete"] = True
    logger.info(f"  CORS Origins: {settings.CORS_ORIGINS}")
    logger.info("=" * 50)
    logger.info("Report Service Ready")
    logger.info("=" * 50)

    yield

    logger.info("Report service shutting down...")
    service_state["startup_complete"] = False


app = FastAPI(
    title=settings.APP_NAME,
    description="Report generation service for MediAssist AI",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Basic health check for load balancers."""
    return {"status": "healthy", "service": "report"}


@app.get("/ready")
async def readiness_check():
    """Detailed readiness check."""
    return {
        "status": "ready" if service_state["startup_complete"] else "starting",
        "services": {
            "database": service_state["database"],
            "storage": service_state["storage"],
        },
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "MediAssist Report Service",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "서버 내부 오류가 발생했습니다.",
            "detail": str(exc) if settings.DEBUG else None,
        },
    )
