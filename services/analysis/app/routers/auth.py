"""Authentication router for login and registration with MongoDB."""
import hashlib
import secrets
import os
from datetime import datetime
from typing import Optional
from bson import ObjectId

from fastapi import APIRouter, HTTPException, status
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError

from ..schemas.auth import (
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    AuthResponse,
    MessageResponse,
)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

# MongoDB configuration
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.environ.get("MONGODB_DB", "mediassist")

# MongoDB client
_mongo_client: Optional[MongoClient] = None
_mongo_db = None


def get_mongodb():
    """Get or initialize MongoDB connection."""
    global _mongo_client, _mongo_db
    if _mongo_client is None:
        try:
            _mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            # Test connection
            _mongo_client.admin.command('ping')
            _mongo_db = _mongo_client[MONGODB_DB]

            # Create unique index on email
            _mongo_db.users.create_index("email", unique=True)

            print(f"[MongoDB] Connected to {MONGODB_URI}, database: {MONGODB_DB}")
        except ConnectionFailure as e:
            print(f"[MongoDB] Connection failed: {e}")
            return None
        except Exception as e:
            print(f"[MongoDB] Error: {e}")
            return None
    return _mongo_db


# In-memory fallback for demo user (when MongoDB is not available)
DEMO_USER = {
    "id": "user_demo_001",
    "email": "demo@mediassist.ai",
    "name": "데모 의사",
    "role": "doctor",
    "password_hash": hashlib.sha256("demo1234".encode()).hexdigest(),
    "created_at": datetime.now(),
}


def hash_password(password: str) -> str:
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash."""
    return hash_password(password) == password_hash


def generate_token() -> str:
    """Generate a random access token."""
    return secrets.token_urlsafe(32)


@router.post("/register", response_model=AuthResponse)
async def register(request: UserRegisterRequest):
    """Register a new user with MongoDB."""
    db = get_mongodb()

    if db is not None:
        try:
            # Create new user document
            user_doc = {
                "email": request.email,
                "password_hash": hash_password(request.password),
                "name": request.name,
                "role": request.role,
                "created_at": datetime.utcnow(),
                "is_active": True,
            }

            # Insert user into MongoDB
            result = db.users.insert_one(user_doc)

            if not result.inserted_id:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="회원가입 처리 중 오류가 발생했습니다.",
                )

            access_token = generate_token()

            return AuthResponse(
                user=UserResponse(
                    id=str(result.inserted_id),
                    email=request.email,
                    name=request.name,
                    role=request.role,
                    created_at=user_doc["created_at"],
                ),
                access_token=access_token,
            )

        except DuplicateKeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="이미 등록된 이메일입니다.",
            )
        except HTTPException:
            raise
        except Exception as e:
            print(f"[Auth] Registration error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"회원가입 처리 중 오류가 발생했습니다: {str(e)}",
            )
    else:
        # Fallback: Demo registration not allowed
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="데이터베이스 연결이 설정되지 않았습니다. MongoDB 설정을 확인해주세요.",
        )


@router.post("/login", response_model=AuthResponse)
async def login(request: UserLoginRequest):
    """Login with email and password."""

    # Check demo user first
    if request.email == DEMO_USER["email"]:
        if verify_password(request.password, DEMO_USER["password_hash"]):
            return AuthResponse(
                user=UserResponse(
                    id=DEMO_USER["id"],
                    email=DEMO_USER["email"],
                    name=DEMO_USER["name"],
                    role=DEMO_USER["role"],
                    created_at=DEMO_USER["created_at"],
                ),
                access_token=generate_token(),
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="이메일 또는 비밀번호가 올바르지 않습니다.",
            )

    # Check MongoDB
    db = get_mongodb()

    if db is not None:
        try:
            # Find user by email
            user_doc = db.users.find_one({"email": request.email})

            if not user_doc:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="이메일 또는 비밀번호가 올바르지 않습니다.",
                )

            if not verify_password(request.password, user_doc["password_hash"]):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="이메일 또는 비밀번호가 올바르지 않습니다.",
                )

            # Update last login
            db.users.update_one(
                {"_id": user_doc["_id"]},
                {"$set": {"last_login": datetime.utcnow()}}
            )

            access_token = generate_token()

            return AuthResponse(
                user=UserResponse(
                    id=str(user_doc["_id"]),
                    email=user_doc["email"],
                    name=user_doc["name"],
                    role=user_doc["role"],
                    created_at=user_doc.get("created_at"),
                ),
                access_token=access_token,
            )

        except HTTPException:
            raise
        except Exception as e:
            print(f"[Auth] Login error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"로그인 처리 중 오류가 발생했습니다: {str(e)}",
            )
    else:
        # MongoDB not available, only demo user works
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="이메일 또는 비밀번호가 올바르지 않습니다.",
        )


@router.post("/logout", response_model=MessageResponse)
async def logout():
    """Logout user (client-side token removal)."""
    return MessageResponse(
        message="로그아웃 되었습니다.",
        success=True,
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user():
    """Get current user info (demo - returns demo user)."""
    return UserResponse(
        id=DEMO_USER["id"],
        email=DEMO_USER["email"],
        name=DEMO_USER["name"],
        role=DEMO_USER["role"],
        created_at=DEMO_USER.get("created_at"),
    )


@router.get("/status")
async def auth_status():
    """Check authentication service status."""
    db = get_mongodb()
    user_count = 0
    if db is not None:
        try:
            user_count = db.users.count_documents({})
        except:
            pass

    return {
        "mongodb_connected": db is not None,
        "mongodb_uri": MONGODB_URI.split("@")[-1] if "@" in MONGODB_URI else MONGODB_URI,
        "database": MONGODB_DB,
        "user_count": user_count,
        "demo_user_available": True,
    }
