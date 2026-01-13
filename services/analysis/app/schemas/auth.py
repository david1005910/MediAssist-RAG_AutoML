"""Authentication schemas."""
import re
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from datetime import datetime


def validate_email(email: str) -> str:
    """Simple email validation."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValueError('유효한 이메일 주소를 입력하세요')
    return email


class UserRegisterRequest(BaseModel):
    """User registration request."""
    email: str
    password: str = Field(..., min_length=8)
    name: str = Field(..., min_length=2)
    role: Literal['admin', 'doctor', 'nurse', 'researcher'] = 'doctor'

    @field_validator('email')
    @classmethod
    def validate_email_field(cls, v: str) -> str:
        return validate_email(v)


class UserLoginRequest(BaseModel):
    """User login request."""
    email: str
    password: str

    @field_validator('email')
    @classmethod
    def validate_email_field(cls, v: str) -> str:
        return validate_email(v)


class UserResponse(BaseModel):
    """User response."""
    id: str
    email: str
    name: str
    role: Literal['admin', 'doctor', 'nurse', 'researcher']
    created_at: datetime | None = None


class AuthResponse(BaseModel):
    """Authentication response."""
    user: UserResponse
    access_token: str
    token_type: str = "bearer"


class MessageResponse(BaseModel):
    """Simple message response."""
    message: str
    success: bool = True
