"""
app/routes/collections.py
==========================
Router Saved Collections StackMatch.

Endpoints:
  GET    /collections                                      – List all collections (metadata)
  POST   /collections                                      – Create new collection
  DELETE /collections/{collection_id}                      – Delete collection
  POST   /collections/{collection_id}/items                – Add item to collection
  DELETE /collections/{collection_id}/items/{question_id}  – Remove item from collection
  GET    /collections/{collection_id}                      – Detail collection with items
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any
import re


from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator

from app.utils.auth import get_current_user
from app.utils.mongodb import get_collections_collection

from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collections", tags=["Collections"])
_COLLECTIONS_FILE = Path("dummy.json")



# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class CollectionCreateRequest(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Nama koleksi tidak boleh kosong.")
        if len(v) > 50:
            raise ValueError("Nama koleksi maksimal 50 karakter.")
        return v


class CollectionItemAddRequest(BaseModel):
    question_id:    str
    question:       str
    answer_preview: str
    tag:            str
    score_fusion:   float
    note:           str | None = None

    @field_validator("question_id", "question", "answer_preview", "tag")
    @classmethod
    def validate_required_strings(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Field string tidak boleh kosong.")
        return v


class CollectionMetadataResponse(BaseModel):
    id:          str
    name:        str
    created_at:  str
    item_count:  int


class CollectionItemResponse(BaseModel):
    question_id:    str
    question:       str
    answer_preview: str
    tag:            str
    score_fusion:   float
    saved_at:       str
    note:           str | None


class CollectionDetailResponse(BaseModel):
    id:          str
    name:        str
    created_at:  str
    items:       list[CollectionItemResponse]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=list[CollectionMetadataResponse],
    summary="List semua koleksi milik user (hanya metadata)",
)
async def list_collections(current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    coll = get_collections_collection()
    cursor = coll.find({"user_id": user_id})
    collections = await cursor.to_list(length=100)
    
    result = []
    for col in collections:
        result.append(
            CollectionMetadataResponse(
                id=col.get("_id"),
                name=col.get("name", ""),
                created_at=col.get("created_at", ""),
                item_count=len(col.get("items", []))
            )
        )
        
    # Urutkan berdasarkan created_at descending (terbaru duluan)
    result.sort(key=lambda x: x.created_at, reverse=True)
    return result


@router.post(
    "",
    response_model=CollectionMetadataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Buat koleksi baru",
)
async def create_collection(
    body: CollectionCreateRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["id"]
    coll = get_collections_collection()
    
    # Batasan maksimal 20 koleksi
    count = await coll.count_documents({"user_id": user_id})
    if count >= 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maksimal koleksi per user adalah 20.",
        )
        
    # Cek jika nama koleksi duplikat (case-insensitive)
    existing = await coll.find_one({
        "user_id": user_id,
        "name": {"$regex": f"^{re.escape(body.name.strip())}$", "$options": "i"}
    })
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Koleksi dengan nama '{body.name}' sudah ada.",
        )
        
    col_id = str(uuid.uuid4())
    new_col = {
        "_id": col_id,
        "user_id": user_id,
        "name": body.name,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "items": []
    }
    
    await coll.insert_one(new_col)
    
    return CollectionMetadataResponse(
        id=col_id,
        name=new_col["name"],
        created_at=new_col["created_at"],
        item_count=0
    )


@router.delete(
    "/{collection_id}",
    summary="Hapus koleksi",
)
async def delete_collection(
    collection_id: str,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["id"]
    coll = get_collections_collection()
    
    res = await coll.delete_one({"_id": collection_id, "user_id": user_id})
    if res.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Koleksi tidak ditemukan.",
        )
        
    return {"message": "Koleksi berhasil dihapus."}


@router.post(
    "/{collection_id}/items",
    response_model=CollectionItemResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Tambah item ke koleksi",
)
async def add_item_to_collection(
    collection_id: str,
    body: CollectionItemAddRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["id"]
    coll = get_collections_collection()
    
    col = await coll.find_one({"_id": collection_id, "user_id": user_id})
    if not col:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Koleksi tidak ditemukan.",
        )
        
    items = col.get("items", [])
    
    # Cek duplikasi item (pertanyaan sudah disimpan)
    if any(item.get("question_id") == body.question_id for item in items):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Pertanyaan ini sudah disimpan di dalam koleksi ini.",
        )
        
    # Batasan maksimal 100 item
    if len(items) >= 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maksimal item dalam satu koleksi adalah 100.",
        )
        
    new_item = {
        "question_id": body.question_id,
        "question": body.question,
        "answer_preview": body.answer_preview,
        "tag": body.tag,
        "score_fusion": body.score_fusion,
        "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        "note": body.note
    }
    
    await coll.update_one(
        {"_id": collection_id, "user_id": user_id},
        {"$push": {"items": new_item}}
    )
    
    return CollectionItemResponse(**new_item)


@router.delete(
    "/{collection_id}/items/{question_id}",
    summary="Hapus satu item dari koleksi",
)
async def remove_item_from_collection(
    collection_id: str,
    question_id: str,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["id"]
    coll = get_collections_collection()
    
    col = await coll.find_one({"_id": collection_id, "user_id": user_id})
    if not col:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Koleksi tidak ditemukan.",
        )
        
    items = col.get("items", [])
    
    # Cari item
    if not any(item.get("question_id") == question_id for item in items):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item tidak ditemukan di dalam koleksi.",
        )
        
    await coll.update_one(
        {"_id": collection_id, "user_id": user_id},
        {"$pull": {"items": {"question_id": question_id}}}
    )
    
    return {"message": "Item berhasil dihapus dari koleksi."}


@router.get(
    "/{collection_id}",
    response_model=CollectionDetailResponse,
    summary="Detail koleksi beserta semua items",
)
async def get_collection_detail(
    collection_id: str,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["id"]
    coll = get_collections_collection()
    
    col = await coll.find_one({"_id": collection_id, "user_id": user_id})
    if not col:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Koleksi tidak ditemukan.",
        )
        
    return CollectionDetailResponse(
        id=col.get("_id"),
        name=col.get("name", ""),
        created_at=col.get("created_at", ""),
        items=[CollectionItemResponse(**item) for item in col.get("items", [])]
    )
