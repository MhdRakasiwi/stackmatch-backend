"""
app/utils/error_handler.py
===========================
Centralized error handling & request logging untuk StackMatch FastAPI.

Ekspor publik:
  register_error_handlers(app: FastAPI) -> None
      Daftarkan semua exception handler dan request-logging middleware ke app.

Format response error seragam:
  {"error": "<pesan>", "code": <status_code>}

Handler per status code:
  400  – validasi input (query kosong/terlalu panjang, tag/rating tidak valid)
  401  – token expired / invalid / blacklisted
  409  – duplikat email atau username
  422  – tipe data salah (override default FastAPI)
  429  – rate limit terlampaui
  500  – pipeline / model error
  503  – FAISS index atau model belum siap

Logging:
  - Setiap request: method, path, status, response time → console + file
  - 5xx: stack trace dicatat di level ERROR
  - File log default: logs/stackmatch.log (buat otomatis)
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import time
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
_LOG_DIR  = Path(__file__).resolve().parents[2] / "logs"
_LOG_FILE = _LOG_DIR / "stackmatch.log"

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT   = "%Y-%m-%d %H:%M:%S"


def _setup_logger() -> logging.Logger:
    """
    Buat logger bernama 'stackmatch' dengan dua handler:
      1. StreamHandler → stdout/stderr (console)
      2. RotatingFileHandler → logs/stackmatch.log (5 MB × 3 backup)
    """
    logger = logging.getLogger("stackmatch")
    if logger.handlers:
        # Sudah dikonfigurasi sebelumnya (misal saat reload uvicorn)
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FMT)

    # Console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File (rotating)
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            _LOG_FILE,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as exc:
        logger.warning("Gagal membuat file log handler: %s", exc)

    logger.propagate = False
    return logger


logger = _setup_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_response(
    status_code: int,
    message: str,
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    """Buat JSONResponse dengan format error seragam."""
    body: dict[str, Any] = {"error": message, "code": status_code}
    if extra:
        body.update(extra)
    return JSONResponse(status_code=status_code, content=body)


def _extract_detail(exc_detail: Any) -> str:
    """
    Normalkan exc.detail ke string.
    HTTPException.detail bisa berupa str, dict, atau list.
    """
    if isinstance(exc_detail, str):
        return exc_detail
    if isinstance(exc_detail, dict):
        return exc_detail.get("message", str(exc_detail))
    if isinstance(exc_detail, list):
        return "; ".join(str(d) for d in exc_detail)
    return str(exc_detail)


# ---------------------------------------------------------------------------
# Per-status-code messages (override jika detail generik/kosong)
# ---------------------------------------------------------------------------
_STATUS_MESSAGES: dict[int, str] = {
    400: "Permintaan tidak valid.",
    401: "Autentikasi diperlukan.",
    403: "Akses ditolak.",
    404: "Sumber daya tidak ditemukan.",
    409: "Konflik data – resource sudah ada.",
    422: "Format data tidak sesuai.",
    429: "Terlalu banyak permintaan. Coba lagi beberapa saat.",
    500: "Terjadi kesalahan internal server.",
    503: "Layanan belum siap. Coba beberapa saat lagi.",
}

# Pesan khusus berdasarkan kata kunci di detail (untuk 401)
_AUTH_KEYWORD_MESSAGES: dict[str, str] = {
    "expired":     "Token kedaluwarsa, silakan refresh token.",
    "kadaluarsa":  "Token kedaluwarsa, silakan refresh token.",
    "blacklist":   "Token sudah dicabut (logout). Silakan login ulang.",
    "dicabut":     "Token sudah dicabut (logout). Silakan login ulang.",
    "invalid":     "Token tidak valid.",
    "tidak valid": "Token tidak valid.",
}


def _map_401_message(detail: str) -> str:
    """Kembalikan pesan 401 yang lebih deskriptif berdasarkan kata kunci."""
    detail_lower = detail.lower()
    for keyword, message in _AUTH_KEYWORD_MESSAGES.items():
        if keyword in detail_lower:
            return message
    return detail or _STATUS_MESSAGES[401]


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

async def _handle_400(request: Request, exc: HTTPException) -> JSONResponse:
    """400 Bad Request – validasi input (query, tag, rating, dll.)."""
    detail = _extract_detail(exc.detail)
    logger.info("400 Bad Request | %s %s | %s", request.method, request.url.path, detail)
    extra = None
    # Jika detail adalah dict (misal dari /search yang sertakan valid_tags)
    if isinstance(exc.detail, dict) and exc.detail.get("valid_tags"):
        extra = {"valid_tags": exc.detail["valid_tags"]}
        detail = exc.detail.get("message", detail)
    return _error_response(400, detail, extra)


async def _handle_401(request: Request, exc: HTTPException) -> JSONResponse:
    """401 Unauthorized – token expired / invalid / blacklisted."""
    detail  = _extract_detail(exc.detail)
    message = _map_401_message(detail)
    logger.info("401 Unauthorized | %s %s | %s", request.method, request.url.path, message)
    return _error_response(401, message)


async def _handle_409(request: Request, exc: HTTPException) -> JSONResponse:
    """409 Conflict – email atau username sudah digunakan."""
    detail = _extract_detail(exc.detail)
    logger.info("409 Conflict | %s %s | %s", request.method, request.url.path, detail)
    return _error_response(409, detail or _STATUS_MESSAGES[409])


async def _handle_422(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 Unprocessable Entity – tipe data tidak sesuai (override default FastAPI)."""
    errors = exc.errors()
    parts: list[str] = []
    for e in errors:
        loc  = " → ".join(str(l) for l in e.get("loc", []) if l != "body")
        msg  = e.get("msg", "")
        inp  = e.get("input", "")
        # Pesan user-friendly per tipe error
        if "missing" in msg.lower():
            parts.append(f"Field '{loc}' wajib diisi.")
        elif "string" in msg.lower() and "pattern" in msg.lower():
            parts.append(f"Field '{loc}' format tidak sesuai.")
        elif "int" in msg.lower() or "integer" in msg.lower():
            parts.append(f"Field '{loc}' harus berupa bilangan bulat, diterima: {inp!r}.")
        elif "float" in msg.lower():
            parts.append(f"Field '{loc}' harus berupa bilangan desimal, diterima: {inp!r}.")
        elif "bool" in msg.lower():
            parts.append(f"Field '{loc}' harus berupa boolean (true/false).")
        else:
            parts.append(f"Field '{loc}': {msg}.")

    message = " | ".join(parts) if parts else "Format data tidak sesuai."
    logger.info("422 Validation | %s %s | %s", request.method, request.url.path, message)
    return _error_response(422, message)


async def _handle_429(request: Request, exc: HTTPException) -> JSONResponse:
    """429 Too Many Requests – rate limit auth."""
    detail = _extract_detail(exc.detail)
    logger.warning("429 Rate Limit | %s %s | IP: %s", request.method, request.url.path,
                   request.client.host if request.client else "unknown")
    return _error_response(429, detail or _STATUS_MESSAGES[429])


async def _handle_500(request: Request, exc: Exception) -> JSONResponse:
    """
    500 Internal Server Error – pipeline / model error tak terduga.
    Log stack trace secara internal; kembalikan pesan aman ke klien.
    """
    tb = traceback.format_exc()
    logger.error(
        "500 Internal Error | %s %s\n%s",
        request.method, request.url.path, tb,
    )
    # Deteksi apakah ini model/pipeline error
    exc_str = str(exc).lower()
    if any(kw in exc_str for kw in ("model", "pipeline", "faiss", "vectorizer", "sbert")):
        client_msg = "Model tidak tersedia atau pipeline mengalami error."
    else:
        client_msg = "Terjadi kesalahan internal. Tim kami sedang menangani masalah ini."
    return _error_response(500, client_msg)


async def _handle_503(request: Request, exc: HTTPException) -> JSONResponse:
    """503 Service Unavailable – FAISS index atau model belum siap."""
    detail = _extract_detail(exc.detail)
    logger.warning("503 Service Unavailable | %s %s | %s",
                   request.method, request.url.path, detail)
    return _error_response(503, detail or _STATUS_MESSAGES[503])


async def _handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Catch-all untuk HTTPException yang tidak ditangani handler khusus di atas.
    Meneruskan detail apa adanya dengan format seragam.
    """
    detail = _extract_detail(exc.detail)
    code   = exc.status_code
    if code >= 500:
        logger.error("%d Error | %s %s | %s", code, request.method, request.url.path, detail)
    else:
        logger.info("%d | %s %s | %s", code, request.method, request.url.path, detail)
    return _error_response(code, detail or _STATUS_MESSAGES.get(code, "Error."))


async def _handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """Fallback untuk exception Python yang tidak dikompensasi HTTPException."""
    return await _handle_500(request, exc)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log setiap HTTP request ke logger 'stackmatch':
      METHOD /path | status=<code> | <elapsed>ms
    """

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            elapsed  = (time.perf_counter() - start) * 1000
            _log_request(request, response.status_code, elapsed)
            return response
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            _log_request(request, 500, elapsed)
            raise exc


def _log_request(request: Request, status_code: int, elapsed_ms: float) -> None:
    ip    = _get_client_ip(request)
    entry = (
        f"{request.method} {request.url.path} | "
        f"status={status_code} | "
        f"{elapsed_ms:.1f}ms | "
        f"ip={ip}"
    )
    if status_code >= 500:
        logger.error("REQUEST | %s", entry)
    elif status_code >= 400:
        logger.warning("REQUEST | %s", entry)
    else:
        logger.info("REQUEST | %s", entry)


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# Registrar utama – dipanggil dari main.py
# ---------------------------------------------------------------------------

def register_error_handlers(app: FastAPI) -> None:
    """
    Daftarkan semua exception handler dan logging middleware ke FastAPI app.

    Dipanggil sekali dari main.py setelah `app = FastAPI(...)`.

    Handler yang didaftarkan (per status code / exception type):
      RequestValidationError → 422 (override default FastAPI)
      HTTPException 400      → pesan validasi + optional valid_tags
      HTTPException 401      → pesan token kontekstual
      HTTPException 409      → konflik duplikat resource
      HTTPException 429      → rate limit
      HTTPException 503      → model/index belum siap
      HTTPException (lain)   → catch-all dengan format seragam
      Exception (Python)     → 500 dengan logging stack trace
    """
    # Middleware logging (ditambah paling akhir agar jadi lapisan paling luar)
    app.add_middleware(RequestLoggingMiddleware)

    # ── RequestValidationError (422) ─────────────────────────────────────────
    app.add_exception_handler(RequestValidationError, _handle_422)  # type: ignore[arg-type]

    # ── Status-code-specific HTTP handlers ───────────────────────────────────
    # Karena FastAPI tidak mendukung filter per status_code secara native,
    # kita daftarkan satu handler untuk HTTPException dan dispatch di dalamnya.
    @app.exception_handler(HTTPException)
    async def _dispatching_http_handler(request: Request, exc: HTTPException) -> JSONResponse:
        dispatch_map = {
            400: _handle_400,
            401: _handle_401,
            409: _handle_409,
            429: _handle_429,
            503: _handle_503,
        }
        handler = dispatch_map.get(exc.status_code)
        if handler:
            return await handler(request, exc)
        if exc.status_code == 500:
            return await _handle_500(request, exc)
        return await _handle_http_exception(request, exc)

    # ── Catch-all Python Exception → 500 ─────────────────────────────────────
    app.add_exception_handler(Exception, _handle_unhandled_exception)  # type: ignore[arg-type]

    logger.info("Error handlers & request logging middleware terdaftar.")
