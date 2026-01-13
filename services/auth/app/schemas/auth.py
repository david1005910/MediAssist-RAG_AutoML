"""Authentication schemas."""
from pydantic import BaseModel
from typing import Optional, List


class UserRegister(BaseModel):
    """User registration request."""
    email: str
    password: str
    name: str
    role: str = "doctor"


class UserLogin(BaseModel):
    """User login request."""
    email: str
    password: str


class Token(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    user: "UserResponse"


class UserResponse(BaseModel):
    """User response without sensitive data."""
    id: str
    email: str
    name: str
    role: str


class UserInDB(BaseModel):
    """User stored in database."""
    id: str
    email: str
    name: str
    role: str
    hashed_password: str


Token.model_rebuild()
