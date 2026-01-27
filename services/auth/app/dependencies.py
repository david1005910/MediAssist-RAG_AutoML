from typing import Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
):
    """Validate JWT token and return current user."""
    # TODO: Implement JWT validation
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not implemented",
    )
