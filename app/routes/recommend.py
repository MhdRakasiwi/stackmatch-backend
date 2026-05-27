"""
app/routes/recommend.py
========================
Router rekomendasi & search StackMatch.

Endpoints (diprefiks /api/v1 oleh main.py):
  POST /recommend  – Rekomendasi umum (tanpa filter tag)
  POST /search     – Search dengan filter tag opsional

Auth bersifat opsional di kedua endpoint:
  - Jika header Authorization: Bearer hadir → simpan ke usage history
  - Jika tidak hadir → lanjut tanpa menyimpan riwayat
"""

from __future__ import annotations

import json
import logging
import uuid
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel, field_validator

from app.services.model_service import MODEL_STORE, hybrid_search, generate_query_suggestions
from app.utils.auth import decode_token, get_current_user
from app.utils.mongodb import get_usage_history_collection


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recommend", tags=["Recommend"])

# ---------------------------------------------------------------------------
# Shared HTTPBearer (auto_error=False → tidak raise jika header tidak ada)
# ---------------------------------------------------------------------------
_optional_bearer = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "model_config.json"
_DEFAULT_VALID_TAGS = ["python", "javascript", "java", "kotlin", "android"]

_TOP_N_DEFAULT = 5
_TOP_N_MIN     = 1
_TOP_N_MAX     = 20


def _get_config() -> dict[str, Any]:
    """Ambil config: prioritaskan MODEL_STORE, fallback ke file, lalu default."""
    if MODEL_STORE.get("config"):
        return MODEL_STORE["config"]
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _get_valid_tags() -> list[str]:
    """Ambil daftar tag valid dari config (lowercase)."""
    cfg = _get_config()
    tags = cfg.get("valid_tags", _DEFAULT_VALID_TAGS)
    return [t.lower() for t in tags]


def _normalize_top_n(value: int | None) -> int:
    """Normalisasi top_n: jika di luar [1, 20] → set ke default 5 (bukan error)."""
    if value is None:
        return _TOP_N_DEFAULT
    if value < _TOP_N_MIN or value > _TOP_N_MAX:
        return _TOP_N_DEFAULT
    return value


# ---------------------------------------------------------------------------
# Usage History – MongoDB
# ---------------------------------------------------------------------------

async def _save_to_history(
    user_id: str,
    query: str,
    tag: str | None,
    result_count: int = 0,
) -> None:
    """
    Tambahkan satu entri ke usage history user di MongoDB.
    """
    entry: dict[str, Any] = {
        "_id":          str(uuid.uuid4()),
        "user_id":      user_id,
        "query":        query,
        "tag":          tag,
        "timestamp":    datetime.now(tz=timezone.utc).isoformat(),
        "result_count": result_count,
    }
    try:
        coll = get_usage_history_collection()
        await coll.insert_one(entry)
    except Exception as exc:
        logger.warning("Gagal menyimpan usage history ke MongoDB: %s", exc)


async def get_user_history(user_id: str) -> list[dict[str, Any]]:
    """Ambil semua history user dari MongoDB (urutan terbaru duluan)."""
    try:
        coll = get_usage_history_collection()
        cursor = coll.find({"user_id": user_id}).sort("timestamp", -1).limit(100)
        history = await cursor.to_list(length=100)
        for h in history:
            h["id"] = h.pop("_id")
        return history
    except Exception as exc:
        logger.warning("Gagal mengambil usage history dari MongoDB: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Optional auth helper
# ---------------------------------------------------------------------------
def _try_get_user_id(request: Request) -> str | None:
    """
    Ekstrak user_id dari Bearer token jika ada.
    Return None jika header tidak ada atau token tidak valid (tidak raise error).
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.lower().startswith("bearer "):
        return None
    token = auth_header[7:].strip()
    if not token:
        return None
    try:
        payload = decode_token(token)
        if payload.get("type") != "access":
            return None
        return payload.get("sub")
    except HTTPException:
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class RecommendRequest(BaseModel):
    query: str
    top_n: int | None = None
    lang:  str        = "auto"

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query tidak boleh kosong.")
        if len(v) > 500:
            raise ValueError("Query maksimal 500 karakter.")
        return v


class SearchRequest(BaseModel):
    query:     str
    tag:       str | None  = None
    top_n:     int | None  = None
    lang:      str         = "auto"
    min_score: float       = 0.0

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query tidak boleh kosong.")
        if len(v) > 500:
            raise ValueError("Query maksimal 500 karakter.")
        return v

    @field_validator("min_score")
    @classmethod
    def validate_min_score(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("min_score harus antara 0.0 dan 1.0.")
        return v


class SearchResultItem(BaseModel):
    id:             str
    question:       str
    score:          float
    score_fusion:   float
    score_tfidf:    float
    score_sbert:    float
    tag:            str
    answer_preview: str
    answer_full:    str


class SearchResponse(BaseModel):
    results:          list[SearchResultItem]
    query_translated: str
    tag_applied:      str | None
    total:            int


class SuggestionResponse(BaseModel):
    related_queries: list[str]
    related_tags:    list[str]


class RecommendResponse(BaseModel):
    results:          list[SearchResultItem]
    query_translated: str
    total:            int
    suggestions:      SuggestionResponse


class SearchResponse(BaseModel):
    results:          list[SearchResultItem]
    query_translated: str
    tag_applied:      str | None
    total:            int
    suggestions:      SuggestionResponse


class SynthesizeResultItem(BaseModel):
    question:      str
    answer_full:   str
    score_fusion:  float
    tag:           str


class SynthesizeRequest(BaseModel):
    query:    str
    results:  list[SynthesizeResultItem]
    language: str

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query tidak boleh kosong.")
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("id", "en"):
            raise ValueError("Language harus 'id' atau 'en'.")
        return v

    @field_validator("results")
    @classmethod
    def validate_results(cls, v: list[SynthesizeResultItem]) -> list[SynthesizeResultItem]:
        if not v:
            raise ValueError("Results tidak boleh kosong.")
        if len(v) > 5:
            raise ValueError("Results maksimal 5 item.")
        return v


class SynthesizeResponse(BaseModel):
    synthesis:    str
    key_points:   list[str]
    confidence:   float
    sources_used: int


# ---------------------------------------------------------------------------
# POST /recommend
# ---------------------------------------------------------------------------
@router.post(
    "",                          # prefix /recommend sudah dari router
    summary="Rekomendasi pertanyaan serupa (tanpa filter tag)",
    status_code=status.HTTP_200_OK,
    response_model=RecommendResponse,
)
async def recommend(body: RecommendRequest, request: Request):
    """
    Cari pertanyaan Stack Overflow yang relevan dengan query.

    - Auth **opsional**: jika Bearer token valid disertakan, query akan disimpan ke
      usage history.
    - `top_n` di luar [1, 20] akan otomatis diset ke **5** (bukan error).
    - `lang` saat ini informatif saja; deteksi bahasa otomatis dilakukan oleh pipeline.
    """
    user_id = _try_get_user_id(request)
    user_preference = None
    if user_id:
        from app.services.model_service import get_user_preference_vector
        user_preference = await get_user_preference_vector(user_id)

    top_n = _normalize_top_n(body.top_n)

    # Jalankan pipeline
    try:
        results, query_translated = await _run_search(
            query=body.query,
            tag=None,
            top_n=top_n,
            min_score=0.0,
            user_preference=user_preference,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        )
    except Exception as exc:
        logger.exception("Unexpected error di /recommend")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {exc}",
        )

    # Simpan ke history jika user login
    if user_id:
        await _save_to_history(user_id, body.query, tag=None, result_count=len(results))

    dataset = MODEL_STORE.get("dataset")
    suggestions = generate_query_suggestions(body.query, results, dataset, filtered_tag=None)

    return {
        "results":          results,
        "query_translated": query_translated,
        "total":            len(results),
        "suggestions":      suggestions,
    }


# ---------------------------------------------------------------------------
# POST /search  (router prefix-nya /recommend, tapi path /search terpisah)
# ---------------------------------------------------------------------------
_search_router = APIRouter(prefix="/search", tags=["Search"])


@_search_router.post(
    "",
    summary="Search dengan filter tag opsional",
    status_code=status.HTTP_200_OK,
    response_model=SearchResponse,
)
async def search(body: SearchRequest, request: Request):
    """
    Cari pertanyaan Stack Overflow dengan filter tag opsional.

    - **tag** harus berupa salah satu dari `valid_tags` di `model_config.json`.
      Jika tidak valid, return 400 beserta daftar tag yang diizinkan.
    - `top_n` di luar [1, 20] akan otomatis diset ke **5**.
    - `min_score` harus antara 0.0–1.0.
    - Auth **opsional**: query + tag disimpan ke history jika token valid.
    """
    user_id = _try_get_user_id(request)
    user_preference = None
    if user_id:
        from app.services.model_service import get_user_preference_vector
        user_preference = await get_user_preference_vector(user_id)

    # Validasi tag
    applied_tag: str | None = None
    if body.tag is not None:
        tag_lower   = body.tag.strip().lower()
        valid_tags  = _get_valid_tags()
        if tag_lower not in valid_tags:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": f"Tag '{body.tag}' tidak valid.",
                    "valid_tags": valid_tags,
                },
            )
        applied_tag = tag_lower

    top_n = _normalize_top_n(body.top_n)

    # Jalankan pipeline
    try:
        results, query_translated = await _run_search(
            query=body.query,
            tag=applied_tag,
            top_n=top_n,
            min_score=body.min_score,
            user_preference=user_preference,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        )
    except Exception as exc:
        logger.exception("Unexpected error di /search")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {exc}",
        )

    # Simpan ke history jika user login
    if user_id:
        await _save_to_history(user_id, body.query, tag=applied_tag, result_count=len(results))

    dataset = MODEL_STORE.get("dataset")
    suggestions = generate_query_suggestions(body.query, results, dataset, filtered_tag=applied_tag)

    return {
        "results":          results,
        "query_translated": query_translated,
        "tag_applied":      applied_tag,
        "total":            len(results),
        "suggestions":      suggestions,
    }


# ---------------------------------------------------------------------------
# Rate Limiter per User untuk Synthesize (10 req / 60 detik)
# ---------------------------------------------------------------------------
_SYNTH_RATE_LIMIT       = 10
_SYNTH_RATE_WINDOW_SEC  = 60
_synth_rate_store: dict[str, list[float]] = defaultdict(list)
_synth_rate_lock: Lock = Lock()


def _check_synth_rate_limit(user_id: str) -> None:
    """
    Periksa rate limit per user untuk endpoint synthesize.
    Raise HTTPException 429 jika melebihi batas.
    """
    now = time.monotonic()
    with _synth_rate_lock:
        timestamps = _synth_rate_store[user_id]
        # Hapus timestamp di luar window
        _synth_rate_store[user_id] = [t for t in timestamps if now - t < _SYNTH_RATE_WINDOW_SEC]
        if len(_synth_rate_store[user_id]) >= _SYNTH_RATE_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Batas request AI Answer Synthesis terlampaui. Silakan coba lagi nanti.",
            )
        _synth_rate_store[user_id].append(now)


def _try_repair_json(text: str) -> str:
    """Mencoba memperbaiki JSON yang terpotong di akhir secara kasar agar tidak memicu 500 crash."""
    text = text.strip()
    if not text:
        return text
    if text.count('"') % 2 != 0:
        text += '"'
    if '"key_points"' in text and "[" in text and "]" not in text:
        text += "]"
    open_braces = text.count("{")
    close_braces = text.count("}")
    if open_braces > close_braces:
        text += "}" * (open_braces - close_braces)
    return text

# ---------------------------------------------------------------------------
# POST /recommend/synthesize
# ---------------------------------------------------------------------------
@router.post(
    "/synthesize",
    response_model=SynthesizeResponse,
    summary="Rangkum hasil pencarian menjadi satu jawaban kohesif menggunakan AI",
    status_code=status.HTTP_200_OK,
)
async def synthesize(
    body: SynthesizeRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Sintesis hasil pencarian (maks 5 item) menjadi satu jawaban terangkum.
    
    - Memerlukan **autentikasi**.
    - Rate limit: **10 request/menit per user**.
    - Memanggil Gemini API.
    """
    # 0. Cek Cache di MongoDB
    import re
    cleaned_query = body.query.strip()
    try:
        usage_coll = get_usage_history_collection()
        # Cari riwayat untuk user ini dengan kueri yang sama (case-insensitive) yang memiliki sintesis
        cached_entry = await usage_coll.find_one({
            "user_id": current_user["id"],
            "query": {"$regex": f"^{re.escape(cleaned_query)}$", "$options": "i"},
            "synthesis": {"$exists": True, "$ne": None}
        })
        if cached_entry:
            scores = [res.score_fusion for res in body.results]
            fallback_conf = round(float(sum(scores) / len(scores)), 4) if scores else 0.0
            
            return SynthesizeResponse(
                synthesis=cached_entry["synthesis"],
                key_points=cached_entry.get("key_points", ["Periksa hasil pencarian di bawah untuk informasi selengkapnya."]),
                confidence=cached_entry.get("confidence", fallback_conf),
                sources_used=cached_entry.get("sources_used", len(body.results)),
            )
    except Exception as exc:
        logger.warning("Gagal membaca cache dari MongoDB: %s", exc)

    # 1. Cek Rate Limit
    _check_synth_rate_limit(current_user["id"])

    # 2. Cek API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fitur AI Answer Synthesis dinonaktifkan karena GEMINI_API_KEY belum dikonfigurasi.",
        )

    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # 3. Hitung confidence & sources_used
    scores = [res.score_fusion for res in body.results]
    confidence = round(float(sum(scores) / len(scores)), 4) if scores else 0.0
    sources_used = len(body.results)

    # 4. Bangun user prompt & system instruction
    sources_text = ""
    for idx, res in enumerate(body.results):
        sources_text += f"Source {idx + 1}:\n"
        sources_text += f"Question: {res.question}\n"
        sources_text += f"Answer: {res.answer_full}\n"
        sources_text += f"Tag: {res.tag}\n\n"

    lang_name = "Indonesian" if body.language == "id" else "English"

    user_prompt = (
        f"User Query: {body.query}\n\n"
        f"Based on the following search sources, synthesize a single cohesive answer for the user's query. "
        f"The answer MUST be in {lang_name} language.\n\n"
        f"Sources:\n{sources_text}\n"
        f"Provide your response in JSON format with two keys:\n"
        f"1. 'synthesis': A cohesive summary paragraph of 150-300 words.\n"
        f"2. 'key_points': An array of 3 to 5 key bullet points summarizing the solution.\n"
        f"Do not write anything else outside the JSON block."
    )

    system_instruction = (
        "You are an AI assistant that synthesizes answers based on retrieved documents. "
        "You must output ONLY valid JSON matching this schema:\n"
        "{\n"
        '  "synthesis": "a cohesive summary paragraph of 150-300 words in the requested language",\n'
        '  "key_points": ["point 1", "point 2", "point 3"]\n'
        "}\n"
        "Do not include any explanation or markdown formatting around the JSON, except optionally wrapping it in standard ```json codeblocks."
    )

    # 5. Panggil Gemini API (Menggunakan endpoint dan struktur resmi Gemini 1.5)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={api_key}"
    headers = {
        "content-type": "application/json",
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"{system_instruction}\n\n{user_prompt}"
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json"
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()

        response_data = response.json()
        raw_text = ""

        # Ekstrak text menggunakan struktur standar Gemini API 1.5
        if isinstance(response_data, dict):
            try:
                candidates = response_data.get("candidates", [])
                if candidates and isinstance(candidates, list):
                    content_val = candidates[0].get("content", {})
                    parts = content_val.get("parts", [])
                    if parts and isinstance(parts, list):
                        raw_text = parts[0].get("text", "")
            except Exception as parse_exc:
                logger.warning("Gagal ekstraksi format standar Gemini: %s", parse_exc)

            # Fallbacks untuk struktur non-standar atau modifikasi API gateway
            if not raw_text:
                candidates = response_data.get("candidates")
                if isinstance(candidates, list) and candidates:
                    first_item = candidates[0]
                    if isinstance(first_item, dict):
                        raw_text = first_item.get("content", "") or first_item.get("output", "")
                        if isinstance(raw_text, dict):
                            parts = raw_text.get("parts", [])
                            if parts and isinstance(parts, list):
                                raw_text = parts[0].get("text", "")
                    elif isinstance(first_item, str):
                        raw_text = first_item

            if not raw_text:
                output = response_data.get("output")
                if isinstance(output, dict):
                    raw_text = output.get("content", "") or output.get("text", "")
                elif isinstance(output, str):
                    raw_text = output

            if not raw_text and response_data.get("content"):
                content_value = response_data.get("content")
                if isinstance(content_value, str):
                    raw_text = content_value
                elif isinstance(content_value, list) and content_value:
                    first_item = content_value[0]
                    if isinstance(first_item, str):
                        raw_text = first_item
                    elif isinstance(first_item, dict):
                        raw_text = first_item.get("text", "") or first_item.get("content", "")

        raw_text = (raw_text or "").strip()
        if not raw_text:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Gemini API returned empty synthesis text.",
            )

        cleaned_text = raw_text
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        parsed_json = {}
        try:
            parsed_json = json.loads(cleaned_text)
        except json.JSONDecodeError:
            try:
                # Coba perbaiki JSON jika terpotong
                repaired_text = _try_repair_json(cleaned_text)
                parsed_json = json.loads(repaired_text)
            except Exception:
                pass

        synthesis = parsed_json.get("synthesis", "")
        key_points = parsed_json.get("key_points", [])

        # Sediakan fallback yang aman jika data tidak lengkap atau parsing gagal
        if not synthesis:
            synthesis = "AI tidak dapat menyelesaikan penyusunan jawaban karena batasan keamanan Google."
        if not isinstance(key_points, list):
            key_points = ["Periksa hasil pencarian di bawah untuk informasi selengkapnya."]

    except httpx.TimeoutException as exc:
        logger.error("Gemini API timeout: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Timeout saat menghubungi layanan AI synthesis (Gemini).",
        )
    except httpx.HTTPStatusError as exc:
        logger.error("Gemini API returned error status %d: %s", exc.response.status_code, exc.response.text)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini API returned error: {exc.response.status_code}",
        )
    except httpx.RequestError as exc:
        logger.error("Gemini API connection error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Gagal menghubungi layanan AI synthesis (Gemini).",
        )
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Gagal melakukan parsing JSON dari Gemini: %s. Raw text: %r", exc, raw_text)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal mengurai respons dari layanan AI.",
        )
    except Exception as exc:
        logger.exception("Kesalahan tidak terduga pada synthesize: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {exc}",
        )

    # Simpan hasil sintesis ke MongoDB (usage_history)
    try:
        usage_coll = get_usage_history_collection()
        cleaned_query = body.query.strip()
        
        # Cari entri terbaru untuk user ini dengan kueri yang sama (case-insensitive)
        existing_entry = await usage_coll.find_one(
            {
                "user_id": current_user["id"],
                "query": {"$regex": f"^{re.escape(cleaned_query)}$", "$options": "i"}
            },
            sort=[("timestamp", -1)]
        )
        
        if existing_entry:
            await usage_coll.update_one(
                {"_id": existing_entry["_id"]},
                {
                    "$set": {
                        "synthesis": synthesis,
                        "key_points": key_points,
                        "confidence": confidence,
                        "sources_used": sources_used,
                    }
                }
            )
        else:
            # Jika tidak ada riwayat sebelumnya, buat entri riwayat baru yang lengkap
            new_entry = {
                "_id": str(uuid.uuid4()),
                "user_id": current_user["id"],
                "query": body.query,
                "tag": None,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "result_count": sources_used,
                "synthesis": synthesis,
                "key_points": key_points,
                "confidence": confidence,
                "sources_used": sources_used,
            }
            await usage_coll.insert_one(new_entry)
    except Exception as exc:
        logger.warning("Gagal menyimpan cache AI synthesis ke MongoDB: %s", exc)

    return SynthesizeResponse(
        synthesis=synthesis,
        key_points=key_points,
        confidence=confidence,
        sources_used=sources_used,
    )


# ---------------------------------------------------------------------------
# Daftarkan sub-router /search ke dalam router utama /recommend
# (main.py hanya mendeteksi satu 'router' per modul; kita gabungkan di sini)
# ---------------------------------------------------------------------------
router.include_router(_search_router)


# ---------------------------------------------------------------------------
# Helper: jalankan hybrid_search dan kembalikan (results, query_translated)
# ---------------------------------------------------------------------------
async def _run_search(
    query: str,
    tag: str | None,
    top_n: int,
    min_score: float,
    user_preference: dict | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """
    Wrapper tipis untuk hybrid_search.
    Kembalikan (results, query_translated) — query_translated diambil dari
    hasil pertama jika ada (pipeline sudah menerjemahkan secara internal).
    """
    results = await hybrid_search(
        query=query,
        tag=tag,
        top_n=top_n,
        min_score=min_score,
        user_preference=user_preference,
    )

    # Ambil terjemahan query dari internal model_service jika tersedia
    # (model_service tidak mengekspos translated query secara langsung;
    #  kita panggil helper translate-nya sebagai preview untuk response)
    query_translated = _peek_translation(query)

    return results, query_translated


def _peek_translation(query: str) -> str:
    """
    Kembalikan terjemahan query (EN) untuk field query_translated di response.
    Tidak mengganggu pipeline — jika gagal, kembalikan query asli.
    """
    try:
        from app.services.model_service import _is_english, _translate_to_english  # noqa: PLC0415

        if _is_english(query):
            return query
        return _translate_to_english(query)
    except Exception:
        return query
