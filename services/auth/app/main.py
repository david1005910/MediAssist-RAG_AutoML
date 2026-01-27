from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers.auth import router as auth_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    # Startup
    print("Auth service starting...")
    yield
    # Shutdown
    print("Auth service shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Authentication and authorization service for MediAssist AI",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/ready")
async def readiness_check():
    return {"status": "ready"}
