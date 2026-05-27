"""
app/utils/auth.py
=================
Fungsi-fungsi autentikasi StackMatch:
  - hash_password / verify_password   → bcrypt
  - create_access_token               → JWT (exp dari JWT_EXPIRE_HOURS)
  - create_refresh_token              → JWT (exp dari JWT_REFRESH_DAYS)
  - decode_token                      → dict payload, HTTPException 401 jika invalid/expired
  - blacklist_token / is_blacklisted  → in-memory token blacklist
  - get_current_user                  → FastAPI dependency injection

Storage pengguna: data/users.json
  Schema setiap item:
    {
      "id":           str   (UUID4),
      "username":     str,
      "email":        str,
      "password_hash":str,
      "created_at":   str   (ISO 8601),
      "total_queries":int
    }
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import bcrypt
import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
load_dotenv()

SECRET_KEY: str      = os.getenv("SECRET_KEY", "changeme-secret-key")
# Import MongoDB helper
from app.utils.mongodb import get_users_collection
import re

ALGORITHM: str       = "HS256"
ACCESS_EXPIRE_HOURS: int  = int(os.getenv("JWT_EXPIRE_HOURS", "24"))
REFRESH_EXPIRE_DAYS: int  = int(os.getenv("JWT_REFRESH_DAYS",  "7"))


# ---------------------------------------------------------------------------
# In-memory token blacklist (thread-safe)
# ---------------------------------------------------------------------------
_blacklist: set[str] = set()
_blacklist_lock: Lock = Lock()


def blacklist_token(token: str) -> None:
    """Tambahkan token ke blacklist (logout / revoke)."""
    with _blacklist_lock:
        _blacklist.add(token)


def is_blacklisted(token: str) -> bool:
    """Cek apakah token sudah di-blacklist."""
    with _blacklist_lock:
        return token in _blacklist


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """
    Hash password menggunakan bcrypt.

    Returns:
        String bcrypt hash (UTF-8 decoded).
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """
    Verifikasi plain-text password terhadap bcrypt hash.

    Returns:
        True jika cocok, False jika tidak.
    """
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------
def _utcnow() -> datetime:
    """Waktu UTC saat ini (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def create_access_token(user_id: str, username: str) -> str:
    """
    Buat JWT access token.

    Args:
        user_id:  ID unik pengguna.
        username: Nama pengguna.

    Returns:
        JWT string bertipe 'access'.
    """
    now = _utcnow()
    payload: dict[str, Any] = {
        "sub":      user_id,
        "username": username,
        "type":     "access",
        "iat":      now,
        "exp":      now + timedelta(hours=ACCESS_EXPIRE_HOURS),
        "jti":      str(uuid.uuid4()),  # JWT ID unik untuk kebutuhan blacklist granular
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    """
    Buat JWT refresh token (umur lebih panjang, hanya berisi sub).

    Args:
        user_id: ID unik pengguna.

    Returns:
        JWT string bertipe 'refresh'.
    """
    now = _utcnow()
    payload: dict[str, Any] = {
        "sub":  user_id,
        "type": "refresh",
        "iat":  now,
        "exp":  now + timedelta(days=REFRESH_EXPIRE_DAYS),
        "jti":  str(uuid.uuid4()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    """
    Decode dan validasi JWT token.

    Raises:
        HTTPException 401: Jika token kadaluarsa, tidak valid, atau di-blacklist.

    Returns:
        Dict payload JWT.
    """
    if is_blacklisted(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token telah dicabut (logout). Silakan login ulang.",
        )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token telah kadaluarsa. Silakan login ulang.",
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token tidak valid: {exc}",
        )


# ---------------------------------------------------------------------------
# User storage – MongoDB
# ---------------------------------------------------------------------------

async def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    """Cari user berdasarkan ID. Return None jika tidak ditemukan."""
    users_coll = get_users_collection()
    user = await users_coll.find_one({"_id": user_id})
    if user:
        user["id"] = user.pop("_id")
    return user


async def get_user_by_username(username: str) -> dict[str, Any] | None:
    """Cari user berdasarkan username (case-insensitive)."""
    users_coll = get_users_collection()
    user = await users_coll.find_one(
        {"username": {"$regex": f"^{re.escape(username)}$", "$options": "i"}}
    )
    if user:
        user["id"] = user.pop("_id")
    return user


async def get_user_by_email(email: str) -> dict[str, Any] | None:
    """Cari user berdasarkan email (case-insensitive)."""
    users_coll = get_users_collection()
    user = await users_coll.find_one(
        {"email": {"$regex": f"^{re.escape(email)}$", "$options": "i"}}
    )
    if user:
        user["id"] = user.pop("_id")
    return user


async def create_user(username: str, email: str, password: str) -> dict[str, Any]:
    """
    Buat user baru dan simpan ke MongoDB.

    Schema:
      id, username, email, password_hash, created_at, total_queries

    Raises:
        HTTPException 409: Jika username atau email sudah terdaftar.

    Returns:
        Dict user baru (tanpa password_hash).
    """
    # Cek duplikasi
    if await get_user_by_username(username):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username sudah digunakan.",
        )
    if await get_user_by_email(email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email sudah terdaftar.",
        )

    new_user: dict[str, Any] = {
        "_id":           str(uuid.uuid4()),
        "username":      username,
        "email":         email,
        "password_hash": hash_password(password),
        "created_at":    _utcnow().isoformat(),
        "total_queries": 0,
    }
    
    users_coll = get_users_collection()
    await users_coll.insert_one(new_user)
    new_user["id"] = new_user.pop("_id")

    # Kembalikan data tanpa password_hash
    return {k: v for k, v in new_user.items() if k != "password_hash"}


async def increment_query_count(user_id: str) -> None:
    """Tambah total_queries user sebesar 1."""
    users_coll = get_users_collection()
    await users_coll.update_one({"_id": user_id}, {"$inc": {"total_queries": 1}})


# ---------------------------------------------------------------------------
# FastAPI Dependency – get_current_user
# ---------------------------------------------------------------------------
_bearer_scheme = HTTPBearer(auto_error=True)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> dict[str, Any]:
    """
    FastAPI dependency: ekstrak & validasi Bearer token, return data user.

    Usage:
        @router.get("/me")
        async def me(user = Depends(get_current_user)):
            return user

    Raises:
        HTTPException 401: Token invalid/expired/blacklisted.
        HTTPException 404: User tidak ditemukan di storage.
    """
    token   = credentials.credentials
    payload = decode_token(token)

    # Pastikan ini access token, bukan refresh token
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Gunakan access token, bukan refresh token.",
        )

    user_id = payload.get("sub")
    user    = await get_user_by_id(user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User tidak ditemukan.",
        )

    # Kembalikan data user tanpa password_hash
    return {k: v for k, v in user.items() if k != "password_hash"}
