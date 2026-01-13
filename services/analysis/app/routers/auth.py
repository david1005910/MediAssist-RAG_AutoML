"""Authentication router for login and registration with Supabase."""
import hashlib
import secrets
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from supabase import create_client, Client

from ..schemas.auth import (
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    AuthResponse,
    MessageResponse,
)
from ..config import settings

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

# Supabase client
_supabase_client: Optional[Client] = None


def get_supabase() -> Optional[Client]:
    """Get or initialize Supabase client."""
    global _supabase_client
    if _supabase_client is None:
        if settings.SUPABASE_URL and settings.SUPABASE_KEY:
            try:
                _supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
                print(f"[Supabase] Connected to {settings.SUPABASE_URL}")
            except Exception as e:
                print(f"[Supabase] Connection failed: {e}")
                return None
    return _supabase_client


# In-memory fallback for demo user (when Supabase is not configured)
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
    """Register a new user."""
    supabase = get_supabase()

    if supabase:
        try:
            # Check if email already exists
            existing = supabase.table("users").select("id").eq("email", request.email).execute()
            if existing.data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="이미 등록된 이메일입니다.",
                )

            # Create new user in Supabase
            user_data = {
                "email": request.email,
                "password_hash": hash_password(request.password),
                "name": request.name,
                "role": request.role,
            }

            result = supabase.table("users").insert(user_data).execute()

            if not result.data:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="회원가입 처리 중 오류가 발생했습니다.",
                )

            new_user = result.data[0]
            access_token = generate_token()

            return AuthResponse(
                user=UserResponse(
                    id=str(new_user["id"]),
                    email=new_user["email"],
                    name=new_user["name"],
                    role=new_user["role"],
                    created_at=new_user.get("created_at"),
                ),
                access_token=access_token,
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
        # Fallback: In-memory registration (for demo without Supabase)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="데이터베이스 연결이 설정되지 않았습니다. Supabase 설정을 확인해주세요.",
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

    # Check Supabase
    supabase = get_supabase()

    if supabase:
        try:
            result = supabase.table("users").select("*").eq("email", request.email).execute()

            if not result.data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="이메일 또는 비밀번호가 올바르지 않습니다.",
                )

            user_data = result.data[0]

            if not verify_password(request.password, user_data["password_hash"]):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="이메일 또는 비밀번호가 올바르지 않습니다.",
                )

            access_token = generate_token()

            return AuthResponse(
                user=UserResponse(
                    id=str(user_data["id"]),
                    email=user_data["email"],
                    name=user_data["name"],
                    role=user_data["role"],
                    created_at=user_data.get("created_at"),
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
    supabase = get_supabase()
    return {
        "supabase_connected": supabase is not None,
        "supabase_url": settings.SUPABASE_URL if settings.SUPABASE_URL else "Not configured",
        "demo_user_available": True,
    }
