"""
StackMatch Backend – FastAPI Entry Point
========================================
- Base path  : /api/v1/
- CORS       : dikonfigurasi via env CORS_ORIGINS (comma-separated)
- Redirect   : HTTP 301 /api/* → /api/v1/*
- Lifespan   : load TF-IDF vectorizer, FAISS index, dan dataset saat startup
- Error fmt  : {"error": "...", "code": <status_code>}
- Routers    : di-load otomatis dari app/routes/
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import re
import time
from contextlib import asynccontextmanager
from typing import Any

# ---------------------------------------------------------------------------
# Optional ML imports — server tetap berjalan meski belum terinstall
# ---------------------------------------------------------------------------
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False
    print("[StackMatch] WARNING: 'faiss' tidak ditemukan — fitur FAISS dinonaktifkan.")
    print("             Install: pip install faiss-cpu")

try:
    import joblib  # type: ignore
    _JOBLIB_AVAILABLE = True
except ImportError:
    joblib = None  # type: ignore
    _JOBLIB_AVAILABLE = False
    print("[StackMatch] WARNING: 'joblib' tidak ditemukan — TF-IDF vectorizer tidak dapat dimuat.")
    print("             Install: pip install scikit-learn")

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.routing import APIRouter
from starlette.middleware.base import BaseHTTPMiddleware

from app.utils.error_handler import register_error_handlers
from app.services.model_service import load_all_models, MODEL_STORE

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------
load_dotenv(override=True)

# Waktu mulai server (monotonic) – dipakai untuk menghitung uptime di /health
_APP_START_TIME: float = time.monotonic()

CORS_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

MODEL_PATH: str = os.getenv("MODEL_PATH", "./models")

# ---------------------------------------------------------------------------
# Paths untuk artefak ML
# ---------------------------------------------------------------------------
VECTORIZER_PATH  = os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl")
FAISS_INDEX_PATH = os.path.join(MODEL_PATH, "faiss_index.faiss")
# File dataset menggunakan nama 'dataframe.parquet' (sesuai file di models/)
DATASET_PATH     = os.path.join(MODEL_PATH, "dataframe.parquet")


# ---------------------------------------------------------------------------
# Lifespan – load model/data saat startup, bersihkan saat shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load semua artefak ML ke MODEL_STORE saat startup.
    MODEL_STORE dipakai oleh seluruh route (data, recommend, analytics).
    """
    print("[StackMatch] Memuat artefak ML ke MODEL_STORE...")
    try:
        load_all_models(MODEL_PATH)
        ds = MODEL_STORE.get("dataset")
        n_rows = len(ds) if ds is not None else 0
        print(f"[StackMatch] Startup selesai. Dataset: {n_rows:,} baris.")
    except Exception as exc:
        print(f"[StackMatch] WARNING: Gagal memuat model: {exc}")

    # Sync ke app.state juga (untuk kompatibilitas backward)
    app.state.vectorizer  = MODEL_STORE.get("vectorizer")
    app.state.faiss_index = MODEL_STORE.get("faiss_index")
    app.state.dataset     = MODEL_STORE.get("dataset")

    yield

    # Shutdown
    print("[StackMatch] Shutdown - membersihkan resource...")
    MODEL_STORE.clear()
    app.state.vectorizer  = None
    app.state.faiss_index = None
    app.state.dataset     = None


# ---------------------------------------------------------------------------
# Aplikasi FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="StackMatch API",
    description="Rekomendasi pertanyaan Stack Overflow menggunakan TF-IDF + FAISS",
    version="1.0.0",
    docs_url="/docs",                    # Swagger UI utama (mudah diakses)
    redoc_url="/redoc",                  # ReDoc UI
    openapi_url="/openapi.json",         # OpenAPI schema (untuk Postman import)
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware: Redirect HTTP 301  /api/<path>  →  /api/v1/<path>
# (hanya jika path BUKAN sudah /api/v1/...)
# ---------------------------------------------------------------------------
_REDIRECT_PATTERN = re.compile(r"^/api/(?!v1/)(.*)$")


class LegacyApiRedirectMiddleware(BaseHTTPMiddleware):
    """301 redirect dari /api/* ke /api/v1/* (kecuali sudah /api/v1/)."""

    async def dispatch(self, request: Request, call_next):
        match = _REDIRECT_PATTERN.match(request.url.path)
        if match:
            new_path = f"/api/v1/{match.group(1)}"
            # Pertahankan query string jika ada
            qs = request.url.query
            location = f"{new_path}?{qs}" if qs else new_path
            return RedirectResponse(url=location, status_code=301)
        return await call_next(request)


app.add_middleware(LegacyApiRedirectMiddleware)

# ---------------------------------------------------------------------------
# Error Handlers & Request Logging
# ---------------------------------------------------------------------------
register_error_handlers(app)

# ---------------------------------------------------------------------------
# Middleware: CORS  ← ditambah TERAKHIR agar menjadi lapisan TERLUAR
# (Starlette middleware stack bersifat LIFO: last-added = outermost = runs first)
# CORSMiddleware harus menjadi yang pertama menangani OPTIONS preflight
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auto-load semua router dari app/routes/
# ---------------------------------------------------------------------------
API_V1_PREFIX = "/api/v1"
import app.routes as routes_pkg  # noqa: E402


def _load_routers() -> None:
    """
    Temukan semua modul di dalam app/routes/, impor, dan daftarkan
    atribut 'router' (APIRouter) yang mereka ekspor ke app.
    """
    pkg_path = routes_pkg.__path__
    pkg_name = routes_pkg.__name__

    for _finder, module_name, _is_pkg in pkgutil.iter_modules(pkg_path):
        full_name = f"{pkg_name}.{module_name}"
        module = importlib.import_module(full_name)

        router: APIRouter | None = getattr(module, "router", None)
        if isinstance(router, APIRouter):
            app.include_router(router, prefix=API_V1_PREFIX)
            print(f"  [OK] Router dimuat: {full_name}")
        else:
            print(f"  [-]  Modul {full_name} tidak memiliki 'router', dilewati")


_load_routers()


# ---------------------------------------------------------------------------
# Alias URL: /api/v1/docs & /api/v1/openapi.json → redirect ke /docs & /openapi.json
# ---------------------------------------------------------------------------
@app.get("/api/v1/docs", include_in_schema=False)
async def redirect_docs():
    """Redirect /api/v1/docs → /docs (Swagger UI)."""
    return RedirectResponse(url="/docs")


@app.get("/api/v1/redoc", include_in_schema=False)
async def redirect_redoc():
    """Redirect /api/v1/redoc → /redoc."""
    return RedirectResponse(url="/redoc")


@app.get("/api/v1/openapi.json", include_in_schema=False)
async def redirect_openapi():
    """Redirect /api/v1/openapi.json → /openapi.json."""
    return RedirectResponse(url="/openapi.json")


# ---------------------------------------------------------------------------
# Health-check root
# ---------------------------------------------------------------------------
@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    """Endpoint dasar untuk memverifikasi API berjalan."""
    return {"status": "ok", "service": "StackMatch API", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Entrypoint (opsional – untuk run langsung dengan `python main.py`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
