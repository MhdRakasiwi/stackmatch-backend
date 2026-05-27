"""
app/routes/auth.py
==================
Router autentikasi StackMatch.

Endpoints (semua diprefiks /api/v1 oleh main.py):
  POST /register  – Daftar akun baru
  POST /login     – Login, dapat access+refresh token  (rate-limit 10 req/menit/IP)
  POST /refresh   – Perbarui access token via refresh token
  POST /logout    – Cabut access token (Auth required)
  GET  /me        – Info user yang sedang login (Auth required)
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from threading import Lock
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, field_validator

from app.utils.auth import (
    ACCESS_EXPIRE_HOURS,
    blacklist_token,
    create_access_token,
    create_refresh_token,
    create_user,
    decode_token,
    get_current_user,
    get_user_by_email,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["Auth"])

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,30}$")
_HAS_LETTER  = re.compile(r"[a-zA-Z]")
_HAS_DIGIT   = re.compile(r"[0-9]")


class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        if not _USERNAME_RE.match(v):
            raise ValueError(
                "Username harus 3–30 karakter dan hanya boleh berisi "
                "huruf, angka, atau underscore."
            )
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password minimal 8 karakter.")
        if not _HAS_LETTER.search(v):
            raise ValueError("Password harus mengandung minimal satu huruf.")
        if not _HAS_DIGIT.search(v):
            raise ValueError("Password harus mengandung minimal satu angka.")
        return v


class RegisterResponse(BaseModel):
    message: str
    user_id: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token:  str
    refresh_token: str
    expires_in:    int          # detik
    token_type:    str = "Bearer"
    user:          dict[str, Any]


class RefreshRequest(BaseModel):
    refresh_token: str


class RefreshResponse(BaseModel):
    access_token: str
    expires_in:   int
    token_type:   str = "Bearer"


# ---------------------------------------------------------------------------
# In-memory rate limiter per IP  (10 req / 60 detik)
# ---------------------------------------------------------------------------
_RATE_LIMIT       = 10          # maks request per window
_RATE_WINDOW_SEC  = 60          # panjang window (detik)

# { ip: [(timestamp, count), ...] }  → disederhanakan menjadi { ip: [timestamps] }
_rate_store: dict[str, list[float]] = defaultdict(list)
_rate_lock: Lock = Lock()


def _check_rate_limit(ip: str) -> None:
    """
    Periksa rate limit untuk satu IP.
    Raise HTTPException 429 jika melebihi batas.
    """
    now = time.monotonic()
    with _rate_lock:
        timestamps = _rate_store[ip]
        # Hapus timestamp di luar window
        _rate_store[ip] = [t for t in timestamps if now - t < _RATE_WINDOW_SEC]
        if len(_rate_store[ip]) >= _RATE_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Terlalu banyak percobaan login. "
                    f"Coba lagi dalam {_RATE_WINDOW_SEC} detik."
                ),
            )
        _rate_store[ip].append(now)


def _get_client_ip(request: Request) -> str:
    """Ambil IP klien, perhatikan proxy forwarding."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# POST /auth/register
# ---------------------------------------------------------------------------
@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    response_model=RegisterResponse,
    summary="Daftar akun baru",
)
async def register(body: RegisterRequest):
    """
    Daftarkan pengguna baru ke StackMatch.

    - **username**: 3–30 karakter, hanya huruf/angka/underscore.
    - **email**: format email valid.
    - **password**: min 8 karakter, harus ada huruf dan angka.

    Return 409 jika username atau email sudah digunakan.
    """
    # create_user sudah handle cek duplikat & raise 409
    new_user = await create_user(
        username=body.username,
        email=body.email,
        password=body.password,
    )
    return RegisterResponse(
        message="Registrasi berhasil",
        user_id=new_user["id"],
    )


# ---------------------------------------------------------------------------
# POST /auth/login
# ---------------------------------------------------------------------------
@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Login dan dapatkan token",
)
async def login(body: LoginRequest, request: Request):
    """
    Login dengan email dan password.

    - Rate limit: **10 request/menit per IP**.
    - Return access token (expire sesuai `JWT_EXPIRE_HOURS`) dan refresh token.
    """
    ip = _get_client_ip(request)
    _check_rate_limit(ip)   # Raise 429 jika melebihi batas

    # Cari user
    user = await get_user_by_email(body.email)
    if user is None or not verify_password(body.password, user.get("password_hash", "")):
        # Pesan generik agar tidak membocorkan info akun
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email atau password salah.",
        )

    access_token  = create_access_token(user["id"], user["username"])
    refresh_token = create_refresh_token(user["id"])

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_EXPIRE_HOURS * 3600,
        user={
            "id":       user["id"],
            "username": user["username"],
            "email":    user["email"],
        },
    )


# ---------------------------------------------------------------------------
# POST /auth/refresh
# ---------------------------------------------------------------------------
@router.post(
    "/refresh",
    response_model=RefreshResponse,
    summary="Perbarui access token",
)
async def refresh(body: RefreshRequest):
    """
    Tukar refresh token yang valid dengan access token baru.

    Tidak memerlukan header Authorization.
    """
    payload = decode_token(body.refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token yang diberikan bukan refresh token.",
        )

    user_id  = payload.get("sub", "")
    username = payload.get("username", "")   # refresh token mungkin tidak ada username

    # Jika username tidak ada di payload refresh, ambil dari storage
    if not username:
        from app.utils.auth import get_user_by_id
        user = await get_user_by_id(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User tidak ditemukan.",
            )
        username = user["username"]

    new_access_token = create_access_token(user_id, username)

    return RefreshResponse(
        access_token=new_access_token,
        expires_in=ACCESS_EXPIRE_HOURS * 3600,
    )


# ---------------------------------------------------------------------------
# POST /auth/logout  (Auth required)
# ---------------------------------------------------------------------------
_bearer_scheme = HTTPBearer(auto_error=True)


@router.post(
    "/logout",
    summary="Logout dan cabut token",
)
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    current_user: dict = Depends(get_current_user),
):
    """
    Logout: tambahkan access token saat ini ke blacklist.

    Membutuhkan header `Authorization: Bearer <access_token>`.
    """
    token = credentials.credentials
    blacklist_token(token)
    return {"message": "Logout berhasil"}


# ---------------------------------------------------------------------------
# GET /auth/me  (Auth required)
# ---------------------------------------------------------------------------
@router.get(
    "/me",
    summary="Info user yang sedang login",
)
async def me(current_user: dict = Depends(get_current_user)):
    """
    Kembalikan data profil user yang sedang terautentikasi.

    Field yang dikembalikan: `id`, `username`, `email`, `created_at`, `total_queries`.
    """
    return {
        "id":            current_user["id"],
        "username":      current_user["username"],
        "email":         current_user["email"],
        "created_at":    current_user["created_at"],
        "total_queries": current_user.get("total_queries", 0),
    }
