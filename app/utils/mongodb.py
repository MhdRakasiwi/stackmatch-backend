"""
app/utils/mongodb.py
===================
Utilitas konektivitas database MongoDB menggunakan motor.motor_asyncio.
"""

from __future__ import annotations

import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "stackmatch")

_client: AsyncIOMotorClient | None = None
_db = None


def get_client() -> AsyncIOMotorClient:
    """Mendapatkan motor AsyncIOMotorClient singleton."""
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URI)
    return _client


def get_database():
    """Mendapatkan database StackMatch singleton."""
    global _db
    if _db is None:
        client = get_client()
        _db = client[MONGO_DB_NAME]
    return _db


def close_connection() -> None:
    """Menutup koneksi MongoDB client."""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None


def get_users_collection():
    return get_database()["users"]


def get_collections_collection():
    return get_database()["collections"]


def get_usage_history_collection():
    return get_database()["usage_history"]


def get_feedback_collection():
    return get_database()["feedback"]
