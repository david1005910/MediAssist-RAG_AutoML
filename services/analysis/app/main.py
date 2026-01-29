import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routers import image, symptom, rag, auth, graph, rna, automl

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Service health state
service_state: Dict[str, Any] = {
    "mongodb": False,
    "redis": False,
    "chromadb": False,
    "neo4j": False,
    "ml_models_loaded": False,
    "startup_complete": False,
}


async def check_mongodb_connection() -> bool:
    """Check MongoDB connection status."""
    try:
        from pymongo import MongoClient
        mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        client.close()
        return True
    except Exception as e:
        logger.warning(f"MongoDB connection check failed: {e}")
        return False


async def check_redis_connection() -> bool:
    """Check Redis connection status."""
    try:
        import redis
        redis_url = settings.REDIS_URL
        r = redis.from_url(redis_url, socket_timeout=3)
        r.ping()
        r.close()
        return True
    except Exception as e:
        logger.warning(f"Redis connection check failed: {e}")
        return False


async def check_neo4j_connection() -> bool:
    """Check Neo4j connection status."""
    try:
        from neo4j import GraphDatabase
        neo4j_uri = settings.NEO4J_URI
        neo4j_user = settings.NEO4J_USER
        neo4j_password = settings.NEO4J_PASSWORD

        if not neo4j_uri or neo4j_uri == "bolt://localhost:7687":
            return False

        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception as e:
        logger.warning(f"Neo4j connection check failed: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    logger.info("=" * 50)
    logger.info("MediAssist Analysis Service Starting...")
    logger.info("=" * 50)

    # Check external service connections
    logger.info("Checking external service connections...")

    service_state["mongodb"] = await check_mongodb_connection()
    logger.info(f"  MongoDB: {'Connected' if service_state['mongodb'] else 'Not available (demo mode)'}")

    service_state["redis"] = await check_redis_connection()
    logger.info(f"  Redis: {'Connected' if service_state['redis'] else 'Not available'}")

    service_state["neo4j"] = await check_neo4j_connection()
    logger.info(f"  Neo4j: {'Connected' if service_state['neo4j'] else 'Not available'}")

    # ML Models - lazy loading, mark as available
    service_state["ml_models_loaded"] = True
    logger.info("  ML Models: Ready (lazy loading enabled)")

    service_state["startup_complete"] = True

    logger.info("=" * 50)
    logger.info(f"Service ready on CORS origins: {settings.CORS_ORIGINS}")
    logger.info("=" * 50)

    yield

    # Shutdown
    logger.info("Shutting down MediAssist Analysis Service...")
    service_state["startup_complete"] = False


app = FastAPI(
    title=settings.APP_NAME,
    description="Medical RAG + ML Analysis API - MediAssist AI",
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

# Include routers
app.include_router(auth.router)
app.include_router(image.router)
app.include_router(symptom.router)
app.include_router(rag.router)
app.include_router(graph.router)
app.include_router(rna.router)
app.include_router(automl.router)


@app.get("/health")
async def health_check():
    """
    Basic health check endpoint for Render/load balancers.
    Returns 200 if service is running.
    """
    return {"status": "healthy", "service": "analysis"}


@app.get("/ready")
async def readiness_check():
    """
    Detailed readiness check with dependency status.
    Used for debugging and monitoring.
    """
    return {
        "status": "ready" if service_state["startup_complete"] else "starting",
        "services": {
            "mongodb": service_state["mongodb"],
            "redis": service_state["redis"],
            "neo4j": service_state["neo4j"],
            "ml_models": service_state["ml_models_loaded"],
        },
        "config": {
            "debug": settings.DEBUG,
            "cors_origins": settings.CORS_ORIGINS,
        },
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "MediAssist Analysis Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "detail": str(exc) if settings.DEBUG else None,
        },
    )
