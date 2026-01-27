from typing import Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
):
    """Validate JWT token and return current user."""
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not implemented",
    )


def get_ml_service():
    """Get ML service instance."""
    # TODO: Implement ML service singleton
    pass


def get_rag_service():
    """Get RAG service instance."""
    # TODO: Implement RAG service singleton
    pass
