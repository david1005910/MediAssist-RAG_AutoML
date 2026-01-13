"""Authentication router."""
from datetime import datetime, timedelta
from typing import Dict
import uuid
import hashlib

from fastapi import APIRouter, HTTPException, status
from jose import jwt

from app.config import settings
from app.schemas.auth import (
    UserRegister,
    UserLogin,
    Token,
    UserResponse,
    UserInDB,
)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash (simple SHA256 for demo)."""
    return get_password_hash(plain_password) == hashed_password


def get_password_hash(password: str) -> str:
    """Hash a password using SHA256 (use bcrypt in production)."""
    return hashlib.sha256(password.encode()).hexdigest()


# In-memory user storage (for demo - use database in production)
users_db: Dict[str, UserInDB] = {}

# Add demo user on startup
demo_user = UserInDB(
    id="demo-user-001",
    email="demo@mediassist.ai",
    name="Demo User",
    role="doctor",
    hashed_password=get_password_hash("demo1234"),
)
users_db[demo_user.email] = demo_user


def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})

    # Use HS256 for simplicity (RS256 requires key pair)
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET, algorithm="HS256")
    return encoded_jwt


@router.post("/register", response_model=Token)
async def register(data: UserRegister):
    """Register a new user."""
    # Check if user already exists
    if data.email in users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 등록된 이메일입니다",
        )

    # Create new user
    user_id = str(uuid.uuid4())
    hashed_password = get_password_hash(data.password)

    new_user = UserInDB(
        id=user_id,
        email=data.email,
        name=data.name,
        role=data.role,
        hashed_password=hashed_password,
    )
    users_db[data.email] = new_user

    # Create access token
    access_token = create_access_token(
        data={"sub": new_user.email, "user_id": new_user.id}
    )

    return Token(
        access_token=access_token,
        user=UserResponse(
            id=new_user.id,
            email=new_user.email,
            name=new_user.name,
            role=new_user.role,
        ),
    )


@router.post("/login", response_model=Token)
async def login(data: UserLogin):
    """Authenticate user and return access token."""
    # Find user
    user = users_db.get(data.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="이메일 또는 비밀번호가 올바르지 않습니다",
        )

    # Verify password
    if not verify_password(data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="이메일 또는 비밀번호가 올바르지 않습니다",
        )

    # Create access token
    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id}
    )

    return Token(
        access_token=access_token,
        user=UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            role=user.role,
        ),
    )


@router.post("/logout")
async def logout():
    """Logout user (client should discard token)."""
    return {"message": "로그아웃 되었습니다"}


@router.get("/me", response_model=UserResponse)
async def get_current_user():
    """Get current user info (simplified - no token validation for demo)."""
    # In production, this would validate the JWT token and return the user
    # For demo, return the demo user
    return UserResponse(
        id=demo_user.id,
        email=demo_user.email,
        name=demo_user.name,
        role=demo_user.role,
    )
