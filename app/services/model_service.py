"""
app/services/model_service.py
==============================
Pipeline ML StackMatch — TF-IDF + SBERT + FAISS hybrid search.

Ekspor publik:
  load_all_models(model_path: str | Path) -> dict
      Load semua artefak ke MODEL_STORE dan kembalikan referensinya.

  hybrid_search(
      query:     str,
      tag:       str | None = None,
      top_n:     int        = 5,
      min_score: float      = 0.0,
  ) -> list[dict]
      Jalankan hybrid search dan kembalikan list hasil ranking.

MODEL_STORE keys:
  vectorizer        – sklearn TfidfVectorizer
  tfidf_matrix      – scipy sparse matrix  (n_docs × vocab)
  faiss_index       – faiss.Index  (semua tag)
  faiss_kotlin      – faiss.Index  (khusus tag kotlin)
  faiss_kotlin_indices – np.ndarray  (peta baris ke dataset global)
  dataset           – pandas DataFrame
  sbert             – SentenceTransformer
  config            – dict  (dari model_config.json)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional ML imports — graceful fallback jika belum terinstall
# ---------------------------------------------------------------------------
try:
    import faiss  # type: ignore
    _FAISS_OK = True
except ImportError:
    faiss = None  # type: ignore
    _FAISS_OK = False
    logging.getLogger(__name__).warning(
        "'faiss' tidak terinstall — FAISS search dinonaktifkan. "
        "Install: pip install faiss-cpu"
    )

try:
    import joblib  # type: ignore
    _JOBLIB_OK = True
except ImportError:
    joblib = None  # type: ignore
    _JOBLIB_OK = False

try:
    import numpy as np  # type: ignore
    _NUMPY_OK = True
except ImportError:
    np = None  # type: ignore
    _NUMPY_OK = False

try:
    import pandas as pd  # type: ignore
    _PANDAS_OK = True
except ImportError:
    pd = None  # type: ignore
    _PANDAS_OK = False

try:
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    _SKLEARN_OK = True
except ImportError:
    cosine_similarity = None  # type: ignore
    _SKLEARN_OK = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global artefak store
# ---------------------------------------------------------------------------
MODEL_STORE: dict[str, Any] = {}

# Lokasi model_config.json  (di root project, dua level atas file ini)
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "model_config.json"

# ---------------------------------------------------------------------------
# Config defaults (dipakai jika model_config.json tidak ada)
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG: dict[str, Any] = {
    "alpha": 0.4,
    "beta": 0.5,
    "gamma": 0.1,
    "top_n_default": 5,
    "min_score_default": 0.0,
    "sbert_model": "all-MiniLM-L6-v2",
    "faiss_candidates_multiplier": 3,
    "translation": {"source_lang": "auto", "target_lang": "en"},
}


def _load_config() -> dict[str, Any]:
    """Baca model_config.json, fallback ke default jika tidak ada."""
    if _CONFIG_PATH.exists():
        try:
            with _CONFIG_PATH.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
                logger.info("model_config.json dimuat dari %s", _CONFIG_PATH)
                return cfg
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Gagal membaca model_config.json: %s – pakai default", exc)
    return _DEFAULT_CONFIG.copy()


# ---------------------------------------------------------------------------
# Loader utama
# ---------------------------------------------------------------------------
def load_all_models(model_path: str | Path) -> dict[str, Any]:
    """
    Load semua artefak ML ke MODEL_STORE.

    Args:
        model_path: Path direktori yang berisi file model (pkl / faiss / parquet).

    Returns:
        Referensi ke MODEL_STORE (dict).

    File yang dimuat:
        tfidf_vectorizer.pkl       – sklearn TfidfVectorizer
        tfidf_matrix.pkl           – scipy sparse TF-IDF matrix
        faiss_index.faiss          – FAISS index utama
        faiss_kotlin.faiss         – FAISS index khusus kotlin
        faiss_kotlin_indices.pkl   – array pemetaan row kotlin → dataset global
        dataframe.parquet          – pandas DataFrame dataset
        all-MiniLM-L6-v2           – SentenceTransformer (download otomatis)
    """
    base = Path(model_path)
    cfg  = _load_config()

    MODEL_STORE["config"] = cfg

    # ── TF-IDF Vectorizer ────────────────────────────────────────────────────
    _load_artifact(
        key="vectorizer",
        path=base / "tfidf_vectorizer.pkl",
        loader=joblib.load,
        label="TF-IDF Vectorizer",
    )

    # ── TF-IDF Matrix ────────────────────────────────────────────────────────
    _load_artifact(
        key="tfidf_matrix",
        path=base / "tfidf_matrix.pkl",
        loader=joblib.load,
        label="TF-IDF Matrix",
    )

    # ── FAISS Index utama ────────────────────────────────────────────────────
    _load_artifact(
        key="faiss_index",
        path=base / "faiss_index.faiss",
        loader=faiss.read_index,
        label="FAISS Index (main)",
    )

    # ── FAISS Index kotlin ───────────────────────────────────────────────────
    _load_artifact(
        key="faiss_kotlin",
        path=base / "faiss_kotlin.faiss",
        loader=faiss.read_index,
        label="FAISS Index (kotlin)",
    )

    # ── Kotlin global-index mapping ──────────────────────────────────────────
    _load_artifact(
        key="faiss_kotlin_indices",
        path=base / "faiss_kotlin_indices.pkl",
        loader=joblib.load,
        label="FAISS Kotlin Indices",
    )

    # ── Dataset (DataFrame) ──────────────────────────────────────────────────
    _load_artifact(
        key="dataset",
        path=base / "dataframe.parquet",
        loader=pd.read_parquet,
        label="Dataset (parquet)",
    )

    # ── SBERT ────────────────────────────────────────────────────────────────
    sbert_model_name: str = cfg.get("sbert_model", "all-MiniLM-L6-v2")
    MODEL_STORE["sbert"] = _load_sbert_model(sbert_model_name)

    return MODEL_STORE


def _load_sbert_model(model_name: str) -> Any | None:
    """
    Muat SentenceTransformer.

    Cache lokal dicoba lebih dulu supaya startup tidak bergantung pada koneksi
    HuggingFace ketika model sudah pernah terunduh di mesin pengembangan.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        logger.warning("  ⚠  sentence-transformers tidak tersedia: %s", exc)
        return None

    try:
        logger.info("Memuat SBERT model '%s' dari cache lokal...", model_name)
        model = SentenceTransformer(model_name, local_files_only=True)
        logger.info("  ✔  SBERT '%s' dimuat dari cache lokal", model_name)
        return model
    except Exception as local_exc:
        logger.info(
            "SBERT '%s' belum tersedia di cache lokal (%s); coba load online.",
            model_name,
            local_exc,
        )

    try:
        logger.info("Memuat SBERT model '%s' secara online...", model_name)
        model = SentenceTransformer(model_name)
        logger.info("  ✔  SBERT '%s' dimuat", model_name)
        return model
    except Exception as online_exc:
        logger.warning("  ⚠  Gagal memuat SBERT '%s': %s", model_name, online_exc)
        return None


def _load_artifact(
    key: str,
    path: Path,
    loader,
    label: str,
) -> None:
    """Helper: load satu artefak ke MODEL_STORE, log hasilnya."""
    if path.exists():
        try:
            MODEL_STORE[key] = loader(str(path))
            logger.info("  ✔  %s dimuat dari %s", label, path)
        except Exception as exc:
            logger.warning("  ⚠  Gagal memuat %s dari %s: %s", label, path, exc)
            MODEL_STORE[key] = None
    else:
        logger.warning("  ⚠  %s tidak ditemukan di %s – dilewati", label, path)
        MODEL_STORE[key] = None


# ---------------------------------------------------------------------------
# Translation helper
# ---------------------------------------------------------------------------
def _translate_to_english(text: str) -> str:
    """
    Terjemahkan teks ke bahasa Inggris menggunakan deep-translator.
    Fallback ke kamus lokal jika terjemahan online gagal atau tidak mengubah teks.
    """
    translated: str | None = None
    try:
        from deep_translator import GoogleTranslator  # type: ignore
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        if (
            translated
            and translated.strip()
            and translated.strip().lower() != text.strip().lower()
        ):
            return translated.strip()
    except Exception as exc:
        logger.debug("Translasi online gagal: %s", exc)

    local_translation = _local_translate(text)
    if (
        local_translation
        and local_translation.strip()
        and local_translation.strip().lower() != text.strip().lower()
    ):
        logger.debug(
            "Translasi lokal diterapkan: '%s' → '%s'",
            text,
            local_translation,
        )
        return local_translation.strip()

    return text


def _local_translate(text: str) -> str:
    """Terjemahkan teks dengan kamus lokal bila terjemahan online tidak tersedia."""
    if not text or not isinstance(text, str):
        return text

    tokens: list[str] = []
    for raw_token in text.split():
        prefix = ""
        suffix = ""
        token = raw_token

        while token and token[0] in '"\'([{<':
            prefix += token[0]
            token = token[1:]
        while token and token[-1] in "\"'.,;:!?)[]}" :
            suffix = token[-1] + suffix
            token = token[:-1]

        mapped = _LOCAL_TRANSLATION_MAP.get(token.lower(), token)
        tokens.append(f"{prefix}{mapped}{suffix}")

    return " ".join(tokens)


# Kata-kata umum bahasa Indonesia yang sering tercampur dengan Inggris
_ID_MARKERS = frozenset([
    "apa", "bagaimana", "cara", "konek", "koneksi", "mengapa", "kenapa",
    "buat", "bisa", "tidak", "dengan", "dari", "untuk", "pada", "dalam",
    "sebuah", "yang", "ini", "itu", "dan", "atau", "jika", "maka",
    "saya", "kita", "dia", "mereka", "kamu", "kami", "ada", "sudah",
    "belum", "perlu", "harus", "ingin", "mau", "bisa", "boleh",
    "pakai", "gunakan", "simpan", "ambil", "buka", "tutup", "ubah",
    "hapus", "tambah", "cari", "tampil", "kirim", "terima", "masuk",
])

_LOCAL_TRANSLATION_MAP = {
    "koneksi": "connection",
    "konek": "connect",
    "cara": "how to",
    "apa": "what",
    "bagaimana": "how",
    "mengapa": "why",
    "kenapa": "why",
    "database": "database",
    "server": "server",
    "client": "client",
    "klien": "client",
    "sistem": "system",
    "file": "file",
    "folder": "folder",
    "direktori": "directory",
    "instal": "install",
    "install": "install",
    "kode": "code",
    "fungsi": "function",
    "kelas": "class",
    "method": "method",
    "metode": "method",
    "query": "query",
    "hasil": "result",
    "cari": "search",
    "token": "token",
    "login": "login",
    "logout": "logout",
    "sandi": "password",
    "kunci": "key",
    "terhubung": "connected",
    "memperbarui": "update",
}


def _is_english(text: str) -> bool:
    """
    Deteksi apakah teks berbahasa Inggris.
    Menggunakan langdetect + word-level check kata penanda bahasa Indonesia.
    Sengaja dibuat lebih konservatif: lebih baik translate yang tidak perlu
    daripada tidak translate query bahasa Indonesia → TF-IDF score = 0.
    """
    if not text or len(text.strip()) < 3:
        return True

    # Cek apakah ada kata penanda bahasa Indonesia di query
    tokens = set(text.lower().split())
    if tokens & _ID_MARKERS:
        logger.debug("[lang] Terdeteksi kata Indonesia di query '%s' → akan ditranslasi", text)
        return False

    try:
        from langdetect import detect, DetectorFactory  # type: ignore
        DetectorFactory.seed = 0  # hasil deterministik
        lang = detect(text)
        return lang == "en"
    except Exception:
        pass

    # Fallback: heuristik ASCII (kurang akurat untuk bahasa Latin)
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) >= 0.85


# ---------------------------------------------------------------------------
# Hybrid Search
# ---------------------------------------------------------------------------
async def hybrid_search(
    query: str,
    tag: str | None = None,
    top_n: int = 5,
    min_score: float = 0.0,
    user_preference: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Jalankan hybrid search TF-IDF + SBERT + FAISS.

    Args:
        query:     Teks pertanyaan dari user.
        tag:       Filter tag (misal 'kotlin'). None = semua tag.
        top_n:     Jumlah hasil yang dikembalikan.
        min_score: Ambang batas skor fusion minimum (0.0–1.0).

    Returns:
        List of dict dengan key:
          id, question, score_fusion, score_tfidf, score_sbert, tag, answer_preview

    Raises:
        RuntimeError: Jika model belum dimuat (MODEL_STORE kosong).
    """
    if not MODEL_STORE:
        raise RuntimeError(
            "MODEL_STORE kosong. Panggil load_all_models() terlebih dahulu."
        )

    cfg: dict     = MODEL_STORE.get("config") or _DEFAULT_CONFIG
    alpha: float  = float(cfg.get("alpha", 0.4))   # bobot TF-IDF
    beta: float   = float(cfg.get("beta",  0.5))   # bobot SBERT
    gamma: float  = float(cfg.get("gamma", 0.1))   # bobot answer_score
    k_mult: int   = int(cfg.get("faiss_candidates_multiplier", 3))

    # ── 1. Translate query ke Inggris jika bukan bahasa Inggris ─────────────
    translated_query = query
    if not _is_english(query):
        translated_query = _translate_to_english(query)
        logger.info("[translate] '%s' → '%s'", query, translated_query)

    dataset: pd.DataFrame | None = MODEL_STORE.get("dataset")
    if dataset is None or len(dataset) == 0:
        logger.warning("Dataset kosong atau belum dimuat – return []")
        return []

    # ── 2. Pilih FAISS index berdasarkan tag ─────────────────────────────────
    use_kotlin_index = (
        tag is not None
        and tag.lower() == "kotlin"
        and MODEL_STORE.get("faiss_kotlin") is not None
        and MODEL_STORE.get("faiss_kotlin_indices") is not None
    )

    if use_kotlin_index:
        active_index     = MODEL_STORE["faiss_kotlin"]
        kotlin_global_ix = np.asarray(MODEL_STORE["faiss_kotlin_indices"]).flatten()
        active_dataset   = dataset.iloc[kotlin_global_ix]
        # Untuk kotlin, gunakan FAISS index khusus
        tag_filtered_indices = None
    else:
        active_index   = MODEL_STORE.get("faiss_index")
        # Pre-filter dataset by tag sebelum SBERT scoring
        # agar SBERT+FAISS bekerja di ruang dokumen yang relevan
        if tag is not None:
            tag_col = _detect_tag_column(dataset)
            if tag_col:
                mask = dataset[tag_col].astype(str).str.strip().str.lower() == tag.strip().lower()
                tag_filtered_indices = np.where(mask.to_numpy())[0]
                active_dataset = dataset.iloc[tag_filtered_indices]
            else:
                tag_filtered_indices = None
                active_dataset = dataset
        else:
            tag_filtered_indices = None
            active_dataset = dataset

    n_docs = len(active_dataset)
    if n_docs == 0:
        return []

    index_size = (
        int(active_index.ntotal)
        if active_index is not None and hasattr(active_index, "ntotal")
        else n_docs
    )
    if tag_filtered_indices is not None and not use_kotlin_index:
        # FAISS index utama berisi semua tag. Ambil seluruh kandidat supaya
        # dokumen pada tag yang jarang tetap mendapat skor SBERT.
        candidates = index_size
    else:
        candidates = min(top_n * k_mult, n_docs, index_size)  # jumlah kandidat FAISS

    global_to_local = None
    if tag_filtered_indices is not None and not use_kotlin_index:
        global_to_local = {
            int(global_idx): local_idx
            for local_idx, global_idx in enumerate(tag_filtered_indices)
        }

    # ── 3. TF-IDF scoring ────────────────────────────────────────────────────
    tfidf_scores = np.zeros(n_docs, dtype=np.float32)

    vectorizer  = MODEL_STORE.get("vectorizer")
    tfidf_matrix = MODEL_STORE.get("tfidf_matrix")

    if vectorizer is not None and tfidf_matrix is not None:
        try:
            query_vec = vectorizer.transform([translated_query])
            # Slice tfidf_matrix agar sesuai dengan active_dataset
            if use_kotlin_index:
                sub_matrix = tfidf_matrix[kotlin_global_ix]
            elif tag_filtered_indices is not None:
                sub_matrix = tfidf_matrix[tag_filtered_indices]
            else:
                sub_matrix = tfidf_matrix
            raw_scores = cosine_similarity(query_vec, sub_matrix).flatten()
            max_v = raw_scores.max()

            # ── Fallback: jika semua skor TF-IDF = 0 dan belum ditranslasi,
            #    coba translasi ulang (handle query mixed-lang / deteksi gagal)
            if max_v == 0 and translated_query == query:
                fallback_q = _translate_to_english(query)
                if fallback_q.strip().lower() != query.strip().lower():
                    logger.info(
                        "[tfidf-fallback] TF-IDF = 0, retry dengan translasi: '%s' → '%s'",
                        query, fallback_q,
                    )
                    query_vec  = vectorizer.transform([fallback_q])
                    raw_scores = cosine_similarity(query_vec, sub_matrix).flatten()
                    max_v      = raw_scores.max()

            # Normalisasi ke [0, 1]
            tfidf_scores = raw_scores / max_v if max_v > 0 else raw_scores
            logger.debug(
                "[tfidf] query='%s' → max_score=%.4f", translated_query, float(max_v)
            )
        except Exception as exc:
            logger.warning("TF-IDF scoring gagal: %s", exc)

    # ── 4. SBERT + FAISS scoring ─────────────────────────────────────────────
    sbert_scores = np.zeros(n_docs, dtype=np.float32)
    faiss_indices_returned: list[int] = []

    sbert = MODEL_STORE.get("sbert")
    if sbert is not None and active_index is not None and candidates > 0:
        try:
            query_emb = sbert.encode(
                [translated_query],
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)

            distances, indices = active_index.search(query_emb, candidates)
            distances = distances.flatten()
            indices = indices.flatten()

            # Inner product / cosine distance → similarity (klamp ke [0,1])
            sim_scores = np.clip(distances, 0.0, 1.0)

            # Normalisasi FAISS scores
            max_s = sim_scores.max()
            if max_s > 0:
                sim_scores = sim_scores / max_s

            for global_idx, score in zip(indices, sim_scores):
                if global_to_local is not None:
                    local_idx = global_to_local.get(int(global_idx))
                    if local_idx is None:
                        continue
                else:
                    local_idx = int(global_idx)

                if 0 <= local_idx < n_docs:
                    sbert_scores[local_idx] = float(score)
                    faiss_indices_returned.append(int(local_idx))
        except Exception as exc:
            logger.warning("SBERT+FAISS scoring gagal: %s", exc)

    # ── 5. Answer score (dari dataset) ───────────────────────────────────────
    answer_scores = np.zeros(n_docs, dtype=np.float32)
    ans_score_col = None
    for _col in ("answer_score", "AnswerScore", "score", "Score"):
        if _col in active_dataset.columns:
            ans_score_col = _col
            break
    if ans_score_col:
        raw_ans = active_dataset[ans_score_col].fillna(0).to_numpy(dtype=np.float32)
        max_ans = raw_ans.max()
        answer_scores = raw_ans / max_ans if max_ans > 0 else raw_ans

    # ── 6. Fusion score ───────────────────────────────────────────────────────
    fusion_scores = (
        alpha * tfidf_scores
        + beta  * sbert_scores
        + gamma * answer_scores
    )

    # ── 6.5. Terapkan preferensi pengguna jika ada ─────────────────────────────
    if user_preference and "tag_weights" in user_preference:
        tag_weights = user_preference["tag_weights"]
        tag_col = _detect_tag_column(active_dataset)
        if tag_col:
            for idx in range(n_docs):
                row = active_dataset.iloc[idx]
                row_tag_val = row.get(tag_col, "")
                if row_tag_val and not pd.isna(row_tag_val):
                    row_tags = [t.strip().lower() for t in str(row_tag_val).replace(",", "|").split("|") if t.strip()]
                    multipliers = [tag_weights[t] for t in row_tags if t in tag_weights]
                    max_multiplier = max(multipliers) if multipliers else 1.0
                    fusion_scores[idx] *= max_multiplier

    # ── 7. Ranking & filter min_score ────────────────────────────────────────
    ranked_indices = np.argsort(fusion_scores)[::-1]

    results: list[dict[str, Any]] = []
    for idx in ranked_indices:
        if len(results) >= top_n:
            break
        score_f = float(fusion_scores[idx])
        if score_f < min_score:
            break  # sudah terurut descending; tidak perlu lanjut
        row = active_dataset.iloc[int(idx)]
        results.append(_build_result(row, score_f, tfidf_scores[idx], sbert_scores[idx]))

    # Urutkan secara eksplisit dari yang tertinggi ke terendah
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Helper: bangun satu item hasil
# ---------------------------------------------------------------------------
def _build_result(
    row: pd.Series,
    score_fusion: float,
    score_tfidf: float,
    score_sbert: float,
) -> dict[str, Any]:
    """Konversi satu baris DataFrame ke dict hasil yang dikembalikan ke klien."""
    # Cari nilai dari beberapa kemungkinan nama kolom (case-insensitive fallback)
    def _get(*keys: str, default: Any = None) -> Any:
        for k in keys:
            val = row.get(k, None)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                return val
        return default

    # Nama kolom aktual dataset: Title, Tags, AnswerBody, AnswerScore
    question = _get("Title", "title", "question", "question_title", default="")
    tag_val  = _get("Tags", "tag", "tags", "Tag", default="")
    q_id     = _get("id", "Id", "question_id", default="")
    if not q_id and q_id != 0:
        q_id = str(row.name)
    else:
        q_id = str(q_id)

    # Answer preview: ambil 300 karakter pertama dari kolom teks jawaban
    body = _get("AnswerBody", "answer_body", "Body", "body", default="")
    body_str = str(body) if body else ""
    answer_preview = body_str[:300].strip()

    return {
        "id":             q_id,
        "question":       str(question),
        "score":          round(score_fusion, 4),  # Nilai score secara keseluruhan
        "score_fusion":   round(score_fusion, 4),
        "score_tfidf":    round(float(score_tfidf), 4),
        "score_sbert":    round(float(score_sbert), 4),
        "tag":            str(tag_val),
        "answer_preview": answer_preview,
        "answer_full":    body_str,
    }


# ---------------------------------------------------------------------------
# Helper: deteksi nama kolom tag di DataFrame
# ---------------------------------------------------------------------------
def _detect_tag_column(df: pd.DataFrame) -> str | None:
    """Cari nama kolom yang mewakili tag di DataFrame."""
    for candidate in ("tag", "tags", "Tag", "Tags"):
        if candidate in df.columns:
            return candidate
    return None


# ---------------------------------------------------------------------------
# File paths & helpers untuk preferensi pengguna – MongoDB
# ---------------------------------------------------------------------------
_FEEDBACK_FILE = Path("dummy_feedback.json")
_USAGE_FILE = Path("dummy_usage.json")


async def get_user_preference_vector(user_id: str) -> dict[str, Any] | None:
    """
    Hitung vektor preferensi pengguna berdasarkan feedback rating dan riwayat query.
    
    Cold start: Jika total interaksi (feedback + usage history) < 3, kembalikan None.
    """
    from app.utils.mongodb import get_feedback_collection, get_usage_history_collection
    
    feedback_coll = get_feedback_collection()
    usage_coll = get_usage_history_collection()
    
    user_feedbacks = await feedback_coll.find({"user_id": user_id}).to_list(length=1000)
    user_usages = await usage_coll.find({"user_id": user_id}).to_list(length=1000)
    
    total_interactions = len(user_feedbacks) + len(user_usages)
    if total_interactions < 3:
        return None
        
    dataset = MODEL_STORE.get("dataset")
    if dataset is None or len(dataset) == 0:
        return None
        
    tag_col = _detect_tag_column(dataset)
    ans_score_col = None
    for _col in ("AnswerScore", "answer_score", "score", "Score"):
        if _col in dataset.columns:
            ans_score_col = _col
            break

    tag_ratings: dict[str, list[int]] = {}
    high_rating_answer_scores: list[float] = []
    
    def _split_tags(raw_val: Any) -> list[str]:
        if not raw_val or pd.isna(raw_val):
            return []
        return [t.strip().lower() for t in str(raw_val).replace(",", "|").split("|") if t.strip()]

    id_col = None
    for col in ("id", "question_id", "Id"):
        if col in dataset.columns:
            id_col = col
            break

    for fb in user_feedbacks:
        q_id = fb.get("question_id")
        rating = fb.get("rating")
        if q_id is None or rating is None:
            continue
            
        row = None
        if id_col:
            matching_rows = dataset[dataset[id_col].astype(str) == str(q_id)]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
        else:
            try:
                idx_val = int(q_id)
                if idx_val in dataset.index:
                    row = dataset.loc[idx_val]
            except ValueError:
                pass
                
        if row is not None:
            tags = _split_tags(row.get(tag_col)) if tag_col else []
            for tag in tags:
                tag_ratings.setdefault(tag, []).append(int(rating))
                
            if int(rating) >= 4 and ans_score_col:
                ans_score = row.get(ans_score_col)
                if ans_score is not None and not pd.isna(ans_score):
                    high_rating_answer_scores.append(float(ans_score))

    tag_searches: dict[str, int] = {}
    for us in user_usages:
        raw_tag = us.get("tag")
        if raw_tag:
            tags = _split_tags(raw_tag)
            for tag in tags:
                tag_searches[tag] = tag_searches.get(tag, 0) + 1

    total_searches = sum(tag_searches.values())
    all_unique_tags = set(tag_ratings.keys()) | set(tag_searches.keys())
    tag_weights: dict[str, float] = {}
    
    for tag in all_unique_tags:
        if tag in tag_ratings:
            avg_rating = sum(tag_ratings[tag]) / len(tag_ratings[tag])
            feedback_factor = avg_rating / 3.0
        else:
            feedback_factor = 1.0
            
        if total_searches > 0 and tag in tag_searches:
            freq = tag_searches[tag] / total_searches
            frequency_factor = 1.0 + freq * 0.5
        else:
            frequency_factor = 1.0
            
        weight = feedback_factor * frequency_factor
        weight = max(0.5, min(2.0, weight))
        tag_weights[tag] = round(weight, 2)
        
    if high_rating_answer_scores:
        preferred_answer_score_min = float(min(high_rating_answer_scores))
    else:
        preferred_answer_score_min = 0.0
        
    return {
        "tag_weights": tag_weights,
        "preferred_answer_score_min": preferred_answer_score_min
    }


def generate_query_suggestions(
    query: str,
    results: list[dict[str, Any]],
    dataset: Any,
    filtered_tag: str | None = None,
) -> dict[str, Any]:
    """
    Hasilkan query dan tag rekomendasi terkait berdasarkan hasil pencarian.
    Harus berjalan sangat cepat (< 50ms).
    """
    related_queries: list[str] = []
    related_tags: list[str] = []

    # 1. Hitung related_tags (distribusi tag dari semua hasil)
    tag_counts: dict[str, int] = {}
    exclude_tag = filtered_tag.strip().lower() if filtered_tag else None

    for res in results:
        raw_tag = res.get("tag", "")
        if raw_tag:
            tags = [t.strip().lower() for t in raw_tag.replace(",", "|").split("|") if t.strip()]
            for t in tags:
                if exclude_tag and t == exclude_tag:
                    continue
                tag_counts[t] = tag_counts.get(t, 0) + 1

    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    related_tags = [t for t, count in sorted_tags[:3]]

    # 2. Hitung related_queries dari top-3 hasil pencarian
    vectorizer = MODEL_STORE.get("vectorizer")
    top_3_results = results[:3]

    STOPWORDS = {
        "how", "to", "in", "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "of", "for", "with", "on", "at", "by", "from", "about", "this", "that", "it", "its",
        "i", "you", "he", "she", "they", "we", "my", "your", "his", "her", "their", "our",
        "code", "example", "question", "answer", "debug", "error", "problem", "solution",
        "using", "use", "used", "get", "set", "make", "create", "do", "does", "did"
    }

    generated = []

    if vectorizer is not None and top_3_results:
        try:
            if hasattr(vectorizer, "get_feature_names_out"):
                feature_names = vectorizer.get_feature_names_out()
            elif hasattr(vectorizer, "get_feature_names"):
                feature_names = vectorizer.get_feature_names()
            else:
                feature_names = None

            if feature_names is not None:
                for res in top_3_results:
                    # Ambil teks representatif (title + answer)
                    text = f"{res.get('question', '')} {res.get('answer_preview', '')}"
                    vec = vectorizer.transform([text])
                    if vec.nnz > 0:
                        coo = vec.tocoo()
                        sorted_indices = np.argsort(coo.data)[::-1]
                        top_terms = []
                        for idx in sorted_indices:
                            term = str(feature_names[coo.col[idx]]).lower().strip()
                            if term.isalpha() and len(term) >= 3 and term not in STOPWORDS:
                                top_terms.append(term)
                                if len(top_terms) >= 5:
                                    break

                        doc_tags = [t.strip().lower() for t in res.get("tag", "").replace(",", "|").split("|") if t.strip()]
                        doc_tag = doc_tags[0] if doc_tags else "python"

                        for term in top_terms:
                            generated.append(f"how to {term} in {doc_tag}")
                            generated.append(f"{term} example {doc_tag}")
                            generated.append(f"debug {term} {doc_tag}")
        except Exception as exc:
            logger.warning("Gagal membangkitkan query suggestions: %s", exc)

    query_lower = query.lower().strip()
    seen = set()
    for q in generated:
        q_clean = q.lower().strip()
        if q_clean != query_lower and q_clean not in seen:
            seen.add(q_clean)
            related_queries.append(q)
            if len(related_queries) >= 3:
                break

    if len(related_queries) < 3:
        fallbacks = [
            f"how to use {query}",
            f"{query} example",
            f"debug {query}"
        ]
        for f in fallbacks:
            if f.lower().strip() != query_lower and f.lower().strip() not in seen:
                related_queries.append(f)
                if len(related_queries) >= 3:
                    break

    return {
        "related_queries": related_queries[:3],
        "related_tags": related_tags
    }

