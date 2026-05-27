"""
app/routes/data.py
==================
Endpoint pendukung / utility StackMatch.

Endpoints (diprefiks /api/v1 oleh main.py):
  GET  /data/questions   – Daftar pertanyaan dengan paginasi & filter tag
  GET  /data/tags        – Count per tag dari dataset
  GET  /data/stats       – Statistik ringkasan dataset
  GET  /data/random      – N pertanyaan acak (opsional filter tag)
  POST /data/translate   – Terjemahkan teks via deep-translator
  GET  /data/health      – Status server & model
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, field_validator

from app.services.model_service import MODEL_STORE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["Data"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_APP_VERSION  = "1.2.0"
_LIMIT_MAX    = 50
_RANDOM_MAX   = 20

# ---------------------------------------------------------------------------
# Dataset accessor helpers
# ---------------------------------------------------------------------------

def _get_dataset() -> pd.DataFrame:
    """Ambil dataset dari MODEL_STORE. Raise 503 jika belum dimuat."""
    ds: pd.DataFrame | None = MODEL_STORE.get("dataset")
    if ds is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dataset belum dimuat. Server sedang dalam proses startup.",
        )
    return ds


def _col(*candidates: str, df: pd.DataFrame) -> str | None:
    """Cari nama kolom pertama yang ada di DataFrame."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_str(val: Any) -> str:
    """Konversi nilai ke string, menangani NaN/None."""
    if val is None:
        return ""
    try:
        if isinstance(val, float) and pd.isna(val):
            return ""
    except Exception:
        pass
    return str(val)


def _extract_tag(row: pd.Series, tag_col: str | None) -> str:
    """Ekstrak nilai tag dari baris DataFrame."""
    if tag_col is None:
        return ""
    return _safe_str(row.get(tag_col, ""))


def _filter_by_tag(df: pd.DataFrame, tag: str | None, tag_col: str | None) -> pd.DataFrame:
    """Filter DataFrame berdasarkan tag (case-insensitive exact match). Return semua jika tag None."""
    if tag is None or tag_col is None:
        return df
    mask = df[tag_col].astype(str).str.strip().str.lower() == tag.strip().lower()
    return df[mask]


# ---------------------------------------------------------------------------
# GET /data/questions
# ---------------------------------------------------------------------------
@router.get(
    "/questions",
    summary="Daftar pertanyaan dengan paginasi dan filter tag",
)
async def get_questions(
    limit:  int         = Query(default=20, ge=1, le=_LIMIT_MAX, description="Maks baris yang dikembalikan (1–50)"),
    offset: int         = Query(default=0,  ge=0,                description="Offset awal"),
    tag:    str | None  = Query(default=None,                    description="Filter tag (case-insensitive contains)"),
):
    """
    Ambil daftar pertanyaan dari dataset dengan paginasi.

    - `limit` maks **50**, default **20**.
    - `offset` mulai dari **0**.
    - `tag` dicocokkan secara *case-insensitive contains* terhadap kolom tag.
    """
    df = _get_dataset()

    id_col  = _col("id",  "Id", "question_id",                       df=df)
    q_col   = _col("Title", "title", "question", "question_title",    df=df)
    tag_col = _col("Tags", "tag", "tags", "Tag",                      df=df)
    score_col = _col("AnswerScore", "Score", "answer_score", "score", df=df)

    filtered = _filter_by_tag(df, tag, tag_col)
    
    # Urutkan berdasarkan popularitas (skor tertinggi ke terendah) jika kolom skor ditemukan
    if score_col:
        filtered = filtered.sort_values(by=score_col, ascending=False, na_position="last")
        
    total    = len(filtered)
    page     = filtered.iloc[offset: offset + limit]

    questions = [
        {
            "id":       _safe_str(row[id_col])  if id_col  else str(i + offset),
            "question": _safe_str(row[q_col])   if q_col   else "",
            "tag":      _extract_tag(row, tag_col),
            "score":    float(row[score_col])   if score_col and not pd.isna(row[score_col]) else 0.0,
        }
        for i, (_, row) in enumerate(page.iterrows())
    ]

    return {
        "questions": questions,
        "total":     total,
        "limit":     limit,
        "offset":    offset,
    }


# ---------------------------------------------------------------------------
# GET /data/tags
# ---------------------------------------------------------------------------
@router.get(
    "/tags",
    summary="Daftar tag beserta jumlah pertanyaan per tag",
)
async def get_tags():
    """
    Hitung jumlah pertanyaan per tag dari dataset.

    Tag bisa berupa satu nilai atau multi-tag dipisah pipe/koma.
    Setiap tag dihitung secara individual.
    """
    df = _get_dataset()

    tag_col = _col("Tags", "tag", "tags", "Tag", df=df)
    if tag_col is None:
        return {"tags": [], "total_tags": 0}

    # Hitung: satu baris bisa punya multi-tag ("|" atau ",")
    tag_counts: dict[str, int] = {}
    for raw in df[tag_col].dropna().astype(str):
        # Coba split pipe dulu, lalu koma
        parts = [t.strip().lower() for t in raw.replace(",", "|").split("|") if t.strip()]
        for part in parts:
            tag_counts[part] = tag_counts.get(part, 0) + 1

    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    tags_list   = [{"tag": t, "count": c} for t, c in sorted_tags]

    return {
        "tags":       tags_list,
        "total_tags": len(tags_list),
    }


# ---------------------------------------------------------------------------
# GET /data/stats
# ---------------------------------------------------------------------------
@router.get(
    "/stats",
    summary="Statistik ringkasan dataset",
)
async def get_stats():
    """
    Kembalikan statistik umum dataset StackMatch:
    - total_questions, total_tags
    - top_tags (5 tag terbanyak)
    - avg_answer_length (rata-rata panjang karakter kolom body/answer)
    - last_updated (waktu saat ini sebagai proxy)
    """
    df = _get_dataset()

    tag_col    = _col("Tags", "tag", "tags", "Tag",                          df=df)
    body_col   = _col("AnswerBody", "body", "answer_body", "accepted_answer", "answers", df=df)

    total_questions = len(df)

    # Tag stats
    tag_counts: dict[str, int] = {}
    if tag_col:
        for raw in df[tag_col].dropna().astype(str):
            parts = [t.strip().lower() for t in raw.replace(",", "|").split("|") if t.strip()]
            for part in parts:
                tag_counts[part] = tag_counts.get(part, 0) + 1

    total_tags = len(tag_counts)
    top_tags   = [
        {"tag": t, "count": c}
        for t, c in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    # Rata-rata panjang teks jawaban
    avg_answer_length: float = 0.0
    if body_col:
        lengths = df[body_col].dropna().astype(str).str.len()
        avg_answer_length = round(float(lengths.mean()), 2) if len(lengths) > 0 else 0.0

    return {
        "total_questions":   total_questions,
        "total_tags":        total_tags,
        "top_tags":          top_tags,
        "avg_answer_length": avg_answer_length,
        "last_updated":      datetime.now(tz=timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# GET /data/random
# ---------------------------------------------------------------------------
@router.get(
    "/random",
    summary="Ambil N pertanyaan acak (opsional filter tag)",
)
async def get_random(
    n:   int        = Query(default=5, ge=1, le=_RANDOM_MAX, description="Jumlah pertanyaan (1–20)"),
    tag: str | None = Query(default=None,                    description="Filter tag opsional"),
):
    """
    Kembalikan `n` pertanyaan acak dari dataset.

    - `n` maks **20**, default **5**.
    - `tag` dicocokkan secara *case-insensitive contains*.
    - `score_fusion` selalu **null** karena tidak ada query perbandingan.
    """
    df = _get_dataset()

    id_col  = _col("id",  "Id", "question_id",                      df=df)
    q_col   = _col("Title", "title", "question", "question_title",   df=df)
    tag_col = _col("Tags", "tag", "tags", "Tag", "Tags",             df=df)

    pool = _filter_by_tag(df, tag, tag_col)

    if len(pool) == 0:
        return {"questions": []}

    score_col = _col("AnswerScore", "Score", "answer_score", "score", df=df)

    if tag is not None and score_col:
        # Jika memfilter tag (misal: php), ambil n pertanyaan paling populer (skor tertinggi)
        sorted_pool = pool.sort_values(by=score_col, ascending=False, na_position="last")
        sample = sorted_pool.head(n)
    else:
        # Jika tidak memfilter tag, ambil secara acak untuk keperluan eksplorasi umum
        sample = pool.sample(n=min(n, len(pool)), random_state=None)
        if score_col:
            sample = sample.sort_values(by=score_col, ascending=False, na_position="last")

    questions = [
        {
            "id":           _safe_str(row[id_col])  if id_col  else "",
            "question":     _safe_str(row[q_col])   if q_col   else "",
            "tag":          _extract_tag(row, tag_col),
            "score":        float(row[score_col])   if score_col and not pd.isna(row[score_col]) else 0.0,
            "score_fusion": float(row[score_col])   if score_col and not pd.isna(row[score_col]) else None,
        }
        for _, row in sample.iterrows()
    ]

    return {"questions": questions}


# ---------------------------------------------------------------------------
# POST /data/translate
# ---------------------------------------------------------------------------
class TranslateRequest(BaseModel):
    text:   str
    source: str = "auto"
    target: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text tidak boleh kosong.")
        if len(v) > 1000:
            raise ValueError("text maksimal 1000 karakter.")
        return v

    @field_validator("target")
    @classmethod
    def validate_target(cls, v: str) -> str:
        v = v.strip().lower()
        if not v:
            raise ValueError("target bahasa wajib diisi.")
        return v

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        return v.strip().lower() if v else "auto"


@router.post(
    "/translate",
    summary="Terjemahkan teks menggunakan deep-translator",
)
async def translate(body: TranslateRequest):
    """
    Terjemahkan teks dari `source` ke `target`.

    - `source` default **'auto'** (deteksi otomatis).
    - `target` wajib diisi (contoh: `'en'`, `'id'`, `'ja'`).
    - Jika terjemahan gagal, `translated` akan sama dengan `original` (fallback).
    """
    translated     = body.text
    source_detected = body.source
    error_msg: str | None = None

    try:
        from deep_translator import GoogleTranslator  # type: ignore

        result = GoogleTranslator(source=body.source, target=body.target).translate(body.text)
        if result and result.strip():
            translated = result.strip()
        # Deteksi bahasa sumber (best-effort)
        try:
            from deep_translator import single_detection  # type: ignore
            source_detected = single_detection(body.text, api_key=None) or body.source
        except Exception:
            source_detected = body.source
    except Exception as exc:
        logger.warning("Translasi gagal (fallback): %s", exc)
        error_msg = str(exc)

    response: dict[str, Any] = {
        "original":        body.text,
        "translated":      translated,
        "source_detected": source_detected,
        "target":          body.target,
    }
    if error_msg:
        response["warning"] = f"Terjemahan gagal, mengembalikan teks asli. ({error_msg})"

    return response


# ---------------------------------------------------------------------------
# GET /data/health
# ---------------------------------------------------------------------------
@router.get(
    "/health",
    summary="Status server dan model ML",
)
async def health():
    """
    Periksa kondisi server StackMatch.

    - `uptime_seconds`: detik sejak server pertama kali berjalan.
    - `model_loaded`: True jika MODEL_STORE sudah terisi (dataset dimuat).
    - `faiss_index_ready`: True jika FAISS index utama siap dipakai.
    - `version`: versi API saat ini.
    """
    # Hitung uptime dari main._APP_START_TIME
    uptime: float = 0.0
    try:
        import main as _main  # type: ignore
        uptime = round(time.monotonic() - _main._APP_START_TIME, 2)
    except Exception:
        uptime = 0.0

    model_loaded     = bool(MODEL_STORE.get("dataset") is not None)
    faiss_ready      = bool(MODEL_STORE.get("faiss_index") is not None)

    return {
        "status":            "ok",
        "uptime_seconds":    uptime,
        "model_loaded":      model_loaded,
        "faiss_index_ready": faiss_ready,
        "version":           _APP_VERSION,
    }
