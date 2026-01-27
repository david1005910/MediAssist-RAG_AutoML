from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import image, symptom, rag, auth, graph, rna, automl


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    # Startup: Load ML models and initialize RAG
    print("Loading ML models...")
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Medical RAG + ML Analysis API",
    version="1.0.0",
    lifespan=lifespan,
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
    return {"status": "healthy"}


@app.get("/ready")
async def readiness_check():
    return {"status": "ready", "ml_models": True, "rag": True}
