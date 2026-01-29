import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from app.config import settings
from app.routers.auth import router as auth_router

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Service health state
service_state: Dict[str, Any] = {
    "mongodb": False,
    "startup_complete": False,
}


async def check_mongodb_connection() -> bool:
    """Check MongoDB connection status."""
    try:
        client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        client.close()
        return True
    except ConnectionFailure as e:
        logger.warning(f"MongoDB connection failed: {e}")
        return False
    except Exception as e:
        logger.warning(f"MongoDB check error: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    logger.info("=" * 50)
    logger.info("MediAssist Auth Service Starting...")
    logger.info("=" * 50)

    # Check MongoDB connection
    service_state["mongodb"] = await check_mongodb_connection()
    if service_state["mongodb"]:
        logger.info(f"  MongoDB: Connected to {settings.MONGODB_DB}")
    else:
        logger.warning("  MongoDB: Not available - using demo user only")

    service_state["startup_complete"] = True
    logger.info(f"  CORS Origins: {settings.CORS_ORIGINS}")
    logger.info("=" * 50)
    logger.info("Auth Service Ready")
    logger.info("=" * 50)

    yield

    logger.info("Auth service shutting down...")
    service_state["startup_complete"] = False


app = FastAPI(
    title=settings.APP_NAME,
    description="Authentication and authorization service for MediAssist AI",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)


@app.get("/health")
async def health_check():
    """Basic health check for load balancers."""
    return {"status": "healthy", "service": "auth"}


@app.get("/ready")
async def readiness_check():
    """Detailed readiness check."""
    return {
        "status": "ready" if service_state["startup_complete"] else "starting",
        "services": {
            "mongodb": service_state["mongodb"],
        },
        "demo_mode": not service_state["mongodb"],
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "MediAssist Auth Service",
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
