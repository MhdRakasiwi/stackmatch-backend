"""
app/routes/analytics.py
========================
Router analytics StackMatch — semua endpoint memerlukan autentikasi.

Endpoints (diprefiks /api/v1 oleh main.py):
  POST /analytics/feedback              – Kirim feedback rating untuk satu pertanyaan
  GET  /analytics/feedback              – Cek apakah user sudah memberi rating (by question_id)
  GET  /analytics/usage                 – Riwayat query pengguna dengan paginasi

Storage:
  data/feedback.json       – list of feedback dict
  data/usage_history.json  – list of usage history dict (ditulis oleh recommend.py)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, field_validator

from app.services.model_service import MODEL_STORE
from app.utils.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])

from app.utils.mongodb import get_feedback_collection, get_usage_history_collection

_FEEDBACK_FILE = Path("dummy_feedback.json")
_USAGE_FILE = Path("dummy_usage.json")




# ---------------------------------------------------------------------------
# Dataset helper – cek keberadaan question_id
# ---------------------------------------------------------------------------

def _question_exists(question_id: int | str) -> bool:
    """
    Cek apakah question_id ada di dataset yang sudah dimuat.
    Mencocokkan kolom: id, question_id, Id (case-sensitive first match).
    Jika kolom ID tidak ditemukan, mencocokkan dengan index DataFrame.
    """
    ds = MODEL_STORE.get("dataset")
    if ds is None:
        # Dataset belum dimuat → lewati validasi, biarkan lolos
        logger.warning("Dataset belum dimuat, validasi question_id dilewati.")
        return True

    q_id_str = str(question_id)
    id_col_found = False
    for col in ("id", "question_id", "Id"):
        if col in ds.columns:
            id_col_found = True
            if q_id_str in ds[col].astype(str).values:
                return True

    if not id_col_found:
        try:
            val = int(question_id)
            return val in ds.index
        except ValueError:
            return q_id_str in ds.index.astype(str)

    return False


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    question_id: int
    query:       str
    rating:      int
    comment:     str | None = None

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v: int) -> int:
        if not (1 <= v <= 5):
            raise ValueError("Rating harus antara 1 sampai 5.")
        return v

    @field_validator("comment")
    @classmethod
    def validate_comment(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.strip()
            if len(v) > 200:
                raise ValueError("Comment maksimal 200 karakter.")
            return v if v else None
        return None

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query tidak boleh kosong.")
        return v


class BatchFeedbackItem(BaseModel):
    has_rated: bool
    rating:    int | None = None


class BatchFeedbackResponse(BaseModel):
    ratings: dict[str, BatchFeedbackItem]


# ---------------------------------------------------------------------------
# POST /analytics/feedback
# ---------------------------------------------------------------------------

@router.post(
    "/feedback",
    status_code=status.HTTP_201_CREATED,
    summary="Kirim feedback rating untuk sebuah pertanyaan",
)
async def submit_feedback(
    body:         FeedbackRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Simpan penilaian (rating 1–5) dari user terhadap satu pertanyaan hasil rekomendasi.

    - `question_id` harus ada di dataset → 400 jika tidak ditemukan.
    - `rating` harus antara **1–5** → 422 jika di luar range (divalidasi Pydantic).
    - `comment` opsional, maks **200** karakter.
    - Duplikasi (user + question_id yang sama) **diizinkan** — setiap feedback disimpan
      sebagai entri baru sehingga history rating lengkap tersimpan.
    """
    # Validasi keberadaan question_id di dataset
    if not _question_exists(body.question_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"question_id {body.question_id!r} tidak ditemukan di dataset.",
        )

    feedback_id = f"fb_{uuid.uuid4().hex[:12]}"

    new_entry: dict[str, Any] = {
        "_id":         feedback_id,
        "user_id":     current_user["id"],
        "question_id": body.question_id,
        "query":       body.query,
        "rating":      body.rating,
        "comment":     body.comment,
        "timestamp":   datetime.now(tz=timezone.utc).isoformat(),
    }

    coll = get_feedback_collection()
    await coll.insert_one(new_entry)

    return {
        "message":     "Feedback diterima",
        "feedback_id": feedback_id,
    }


# ---------------------------------------------------------------------------
# GET /analytics/feedback/batch
# ---------------------------------------------------------------------------

@router.get(
    "/feedback/batch",
    response_model=BatchFeedbackResponse,
    response_model_exclude_none=True,
    summary="Cek status rating user untuk beberapa pertanyaan sekaligus (maks 50)",
)
async def get_feedback_batch(
    question_ids: str = Query(
        ...,
        description="Daftar ID pertanyaan dipisah koma (maksimal 50 ID), contoh: '1,2,3'",
    ),
    current_user: dict = Depends(get_current_user),
):
    """
    Cek apakah user sudah memberikan rating untuk daftar `question_ids` yang diberikan.
    
    - `question_ids` berupa string dipisah koma.
    - Maksimal **50** ID dalam satu request.
    - Mengembalikan status rating beserta nilainya (jika ada).
    """
    if not question_ids.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Parameter 'question_ids' tidak boleh kosong.",
        )
        
    parts = [p.strip() for p in question_ids.split(",") if p.strip()]
    if not parts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Parameter 'question_ids' tidak berisi ID yang valid.",
        )
        
    if len(parts) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maksimal 50 ID per request.",
        )
        
    parsed_ids: list[int] = []
    for part in parts:
        try:
            parsed_ids.append(int(part))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"ID '{part}' bukan integer yang valid.",
            )
            
    user_id = current_user["id"]
    coll = get_feedback_collection()
    cursor = coll.find({
        "user_id": user_id,
        "question_id": {"$in": parsed_ids}
    })
    all_entries = await cursor.to_list(length=1000)
        
    # Group feedback entries by question_id for this user
    user_feedback: dict[int, list[dict[str, Any]]] = {}
    for entry in all_entries:
        try:
            q_id = int(entry.get("question_id"))
            user_feedback.setdefault(q_id, []).append(entry)
        except (ValueError, TypeError):
            continue
                
    ratings: dict[str, BatchFeedbackItem] = {}
    for q_id in parsed_ids:
        matches = user_feedback.get(q_id, [])
        if not matches:
            ratings[str(q_id)] = BatchFeedbackItem(has_rated=False, rating=None)
        else:
            # Ambil entry terbaru berdasarkan timestamp
            latest = max(matches, key=lambda e: e.get("timestamp", ""))
            ratings[str(q_id)] = BatchFeedbackItem(
                has_rated=True,
                rating=latest.get("rating"),
            )
            
    return BatchFeedbackResponse(ratings=ratings)


# ---------------------------------------------------------------------------
# GET /analytics/feedback?question_id=<int>
# ---------------------------------------------------------------------------

@router.get(
    "/feedback",
    summary="Cek apakah user sudah memberi rating untuk sebuah pertanyaan",
)
async def get_feedback(
    question_id:  int | None = Query(
        default=None,
        description="ID pertanyaan yang ingin dicek (wajib, harus integer).",
    ),
    current_user: dict = Depends(get_current_user),
):
    """
    Kembalikan feedback **terbaru** user ini untuk `question_id` yang diminta.

    - `question_id` **wajib** dan harus berupa integer → 400 jika tidak disertakan.
    - Jika belum pernah memberi rating: `has_rated: false`.
    - Jika sudah: `has_rated: true` beserta `rating`, `comment`, `feedback_id`.
    """
    if question_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Parameter 'question_id' wajib disertakan dan harus berupa integer.",
        )

    user_id = current_user["id"]
    coll = get_feedback_collection()
    
    cursor = coll.find({
        "user_id": user_id,
        "question_id": question_id
    }).sort("timestamp", -1).limit(1)
    matches = await cursor.to_list(length=1)

    if not matches:
        return {
            "question_id": question_id,
            "has_rated":   False,
            "rating":      None,
            "comment":     None,
            "feedback_id": None,
        }

    latest = matches[0]

    return {
        "question_id": question_id,
        "has_rated":   True,
        "rating":      latest.get("rating"),
        "comment":     latest.get("comment"),
        "feedback_id": latest.get("_id"),
    }


# ---------------------------------------------------------------------------
# GET /analytics/usage?limit=20&offset=0
# ---------------------------------------------------------------------------

@router.get(
    "/usage",
    summary="Riwayat query pengguna dengan paginasi",
)
async def get_usage(
    limit:        int  = Query(default=20, ge=1,  le=100, description="Maks item (1–100), default 20"),
    offset:       int  = Query(default=0,  ge=0,          description="Offset awal, default 0"),
    current_user: dict = Depends(get_current_user),
):
    """
    Kembalikan riwayat query yang sudah dilakukan oleh user yang sedang login.

    - Dibaca dari `usage_history` collection dan difilter berdasarkan `user_id`.
    - Diurutkan dari **terbaru** ke terlama.
    - `tag` bernilai **null** jika query berasal dari endpoint `/recommend`.
    - `result_count` adalah jumlah hasil yang dikembalikan saat query dilakukan.
    - Paginasi: `limit` maks **100**, `offset` mulai dari **0**.
    """
    user_id = current_user["id"]
    coll = get_usage_history_collection()
    
    total = await coll.count_documents({"user_id": user_id})
    cursor = coll.find({"user_id": user_id}).sort("timestamp", -1).skip(offset).limit(limit)
    paged = await cursor.to_list(length=limit)

    history = [
        {
            "id":           e.get("_id"),
            "query":        e.get("query"),
            "tag":          e.get("tag"),          # None jika dari /recommend
            "timestamp":    e.get("timestamp"),
            "result_count": e.get("result_count", 0),
            "synthesis":    e.get("synthesis"),     # None jika belum disintesis AI
            "key_points":   e.get("key_points"),
            "confidence":   e.get("confidence"),
            "sources_used": e.get("sources_used"),
        }
        for e in paged
    ]

    return {
        "history": history,
        "total":   total,
        "limit":   limit,
        "offset":  offset,
    }


class PreferenceResponse(BaseModel):
    tag_weights: dict[str, float]
    total_interactions: int
    is_personalized: bool


# ---------------------------------------------------------------------------
# GET /analytics/preferences
# ---------------------------------------------------------------------------

@router.get(
    "/preferences",
    response_model=PreferenceResponse,
    summary="Ambil preferensi bobot tag pencarian pengguna",
)
async def get_preferences(current_user: dict = Depends(get_current_user)):
    """
    Dapatkan preferensi pencarian personal pengguna saat ini.
    """
    user_id = current_user["id"]

    feedback_coll = get_feedback_collection()
    usage_coll = get_usage_history_collection()
    
    feedback_list = await feedback_coll.find({"user_id": user_id}).to_list(length=1000)
    usage_list = await usage_coll.find({"user_id": user_id}).to_list(length=1000)
    total_interactions = len(feedback_list) + len(usage_list)

    from app.services.model_service import get_user_preference_vector
    pref = await get_user_preference_vector(user_id)

    tag_weights = {}
    if pref and "tag_weights" in pref:
        tag_weights = pref["tag_weights"]

    return PreferenceResponse(
        tag_weights=tag_weights,
        total_interactions=total_interactions,
        is_personalized=total_interactions >= 3,
    )


# ---------------------------------------------------------------------------
# DELETE /analytics/preferences
# ---------------------------------------------------------------------------

@router.delete(
    "/preferences",
    summary="Reset (hapus) preferensi bobot tag pencarian pengguna",
)
async def delete_preferences(current_user: dict = Depends(get_current_user)):
    """
    Hapus preferensi personal pengguna dengan menghapus seluruh data feedback rating
    dan riwayat pencarian miliknya.
    """
    user_id = current_user["id"]

    feedback_coll = get_feedback_collection()
    usage_coll = get_usage_history_collection()
    
    await feedback_coll.delete_many({"user_id": user_id})
    await usage_coll.delete_many({"user_id": user_id})

    return {"message": "Preferensi pencarian berhasil direset."}


class TrendingQueryItem(BaseModel):
    query: str
    count: int
    tag: str | None = None


class TagDistributionItem(BaseModel):
    tag: str
    count: int
    percentage: float


class TrendingResponse(BaseModel):
    trending_queries: list[TrendingQueryItem]
    tag_distribution: list[TagDistributionItem]
    total_searches_today: int
    total_searches_week: int
    active_users_today: int


# ---------------------------------------------------------------------------
# GET /analytics/trending
# ---------------------------------------------------------------------------

@router.get(
    "/trending",
    response_model=TrendingResponse,
    summary="Ambil data trending: query populer, distribusi tag, dan statistik penggunaan",
)
async def get_trending():
    """
    Ambil data statistik trending pencarian.
    Akses publik (auth opsional).
    """
    from collections import Counter
    from datetime import timedelta

    usage_coll = get_usage_history_collection()
    all_entries = await usage_coll.find().to_list(length=100000)

    now = datetime.now(tz=timezone.utc)
    total_searches_today = 0
    total_searches_week = 0
    active_users_today = set()

    query_stats = {}
    total_tags_count = Counter()

    for entry in all_entries:
        query_str = entry.get("query", "").strip()
        if not query_str:
            continue

        ts_str = entry.get("timestamp")
        if not ts_str:
            continue

        try:
            if ts_str.endswith("Z"):
                ts_str = ts_str[:-1] + "+00:00"
            entry_time = datetime.fromisoformat(ts_str)
        except Exception:
            continue

        delta = now - entry_time
        is_today = delta <= timedelta(days=1)
        is_week = delta <= timedelta(days=7)

        if is_week:
            total_searches_week += 1
        if is_today:
            total_searches_today += 1
            uid = entry.get("user_id")
            if uid:
                active_users_today.add(str(uid))

        # Group queries case-insensitively
        q_lower = query_str.lower()
        tag = entry.get("tag")

        if q_lower not in query_stats:
            query_stats[q_lower] = {
                "raw_query": query_str,
                "count": 0,
                "tags": Counter()
            }

        query_stats[q_lower]["count"] += 1
        if tag:
            query_stats[q_lower]["tags"][tag] += 1
            total_tags_count[tag] += 1

    # Build trending queries (top 5)
    trending_queries = []
    sorted_queries = sorted(query_stats.values(), key=lambda x: (-x["count"], x["raw_query"]))

    for q_data in sorted_queries[:5]:
        tags_counter = q_data["tags"]
        most_common_tag = None
        if tags_counter:
            most_common_tag = tags_counter.most_common(1)[0][0]

        trending_queries.append(
            TrendingQueryItem(
                query=q_data["raw_query"],
                count=q_data["count"],
                tag=most_common_tag,
            )
        )

    # Build tag distribution
    tag_distribution = []
    total_tag_searches = sum(total_tags_count.values())
    if total_tag_searches > 0:
        sorted_tags = sorted(total_tags_count.items(), key=lambda x: (-x[1], x[0]))
        for tag, count in sorted_tags:
            percentage = round((count / total_tag_searches) * 100, 4)
            tag_distribution.append(
                TagDistributionItem(
                    tag=tag,
                    count=count,
                    percentage=percentage,
                )
            )

    return TrendingResponse(
        trending_queries=trending_queries,
        tag_distribution=tag_distribution,
        total_searches_today=total_searches_today,
        total_searches_week=total_searches_week,
        active_users_today=len(active_users_today),
    )

