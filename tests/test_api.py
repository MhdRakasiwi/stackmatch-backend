"""
tests/test_api.py
==================
Unit test StackMatch Backend — pytest + httpx AsyncClient.

Cakupan test:
  1.  test_register_success         → 201, user_id ada
  2.  test_register_duplicate_email → 409
  3.  test_login_success            → 200, access_token & refresh_token ada
  4.  test_login_wrong_password     → 401
  5.  test_refresh_token            → 200, access_token baru dikembalikan
  6.  test_logout_then_use_token    → setelah logout, request dengan token lama → 401
  7.  test_recommend_valid_query    → 200, results berisi item dengan semua score fields
  8.  test_recommend_empty_query    → 400
  9.  test_search_invalid_tag       → 400
  10. test_questions_pagination     → limit & offset berfungsi, total ada
  11. test_random_query_string      → parameter tag dan n via query string
  12. test_feedback_post_then_get   → has_rated: true setelah POST
  13. test_health                   → model_loaded ada (True jika model dimuat)

Cara menjalankan:
  pip install pytest pytest-asyncio httpx
  pytest tests/test_api.py -v

Catatan:
  - Test didesain agar tetap lulus meskipun model ML belum dimuat
    (MODEL_STORE kosong → endpoint /recommend mungkin kembalikan [] atau 500,
     tapi auth & data endpoints tetap berjalan independen).
  - Setiap test_register_* menggunakan email unik (timestamp-based) agar
    tidak bentrok saat dijalankan berulang.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# ---------------------------------------------------------------------------
# MongoDB Mocking
# ---------------------------------------------------------------------------
import re
import app.utils.mongodb as mongodb_mod

import copy

class MockCursor:
    def __init__(self, data):
        self.data = list(data)

    def sort(self, key, direction=1):
        reverse = (direction == -1)
        self.data.sort(key=lambda x: x.get(key, ""), reverse=reverse)
        return self

    def skip(self, n):
        self.data = self.data[n:]
        return self

    def limit(self, n):
        self.data = self.data[:n]
        return self

    async def to_list(self, length=None):
        res = self.data[:length] if length else self.data
        return [copy.deepcopy(doc) for doc in res]


class MockCollection:
    def __init__(self, name):
        self.name = name
        self.documents = []

    async def find_one(self, query, sort=None):
        results = self._filter(query)
        if not results:
            return None
        if sort:
            for key, direction in reversed(sort):
                reverse = (direction == -1)
                results.sort(key=lambda x: x.get(key, ""), reverse=reverse)
        return copy.deepcopy(results[0])

    def find(self, query=None):
        results = self._filter(query or {})
        return MockCursor(results)

    async def insert_one(self, doc):
        doc_copy = copy.deepcopy(doc)
        if "_id" not in doc_copy:
            doc_copy["_id"] = str(uuid.uuid4())
        self.documents.append(doc_copy)
        return type("InsertResult", (), {"inserted_id": doc_copy["_id"]})()

    async def update_one(self, query, update):
        results = self._filter(query)
        if not results:
            return type("UpdateResult", (), {"matched_count": 0, "modified_count": 0})()
        doc = results[0]

        if "$inc" in update:
            for k, v in update["$inc"].items():
                doc[k] = doc.get(k, 0) + v
        if "$push" in update:
            for k, v in update["$push"].items():
                if k not in doc:
                    doc[k] = []
                doc[k].append(copy.deepcopy(v))
        if "$pull" in update:
            for k, v in update["$pull"].items():
                if k in doc and isinstance(doc[k], list):
                    new_list = []
                    for item in doc[k]:
                        match = True
                        for qk, qv in v.items():
                            if item.get(qk) != qv:
                                match = False
                                break
                        if not match:
                            new_list.append(copy.deepcopy(item))
                    doc[k] = new_list

        return type("UpdateResult", (), {"matched_count": 1, "modified_count": 1})()

    async def delete_one(self, query):
        results = self._filter(query)
        if results:
            self.documents.remove(results[0])
            return type("DeleteResult", (), {"deleted_count": 1})()
        return type("DeleteResult", (), {"deleted_count": 0})()

    async def delete_many(self, query):
        docs = self._filter(query)
        count = len(docs)
        for doc in docs:
            self.documents.remove(doc)
        return type("DeleteResult", (), {"deleted_count": count})()

    async def count_documents(self, query):
        docs = self._filter(query)
        return len(docs)

    def _filter(self, query):
        results = []
        for doc in self.documents:
            match = True
            for k, v in query.items():
                if isinstance(v, dict):
                    if "$regex" in v:
                        pattern = v["$regex"]
                        options = v.get("$options", "")
                        flags = re.IGNORECASE if "i" in options else 0
                        val = doc.get(k, "")
                        if not re.search(pattern, str(val), flags):
                            match = False
                            break
                    elif "$in" in v:
                        val = doc.get(k)
                        if val not in v["$in"]:
                            match = False
                            break
                else:
                    if k == "_id" and doc.get("_id") != v:
                        match = False
                        break
                    elif k != "_id" and doc.get(k) != v:
                        match = False
                        break
            if match:
                results.append(doc)
        return results


mock_users = MockCollection("users")
mock_collections = MockCollection("collections")
mock_usage_history = MockCollection("usage_history")
mock_feedback = MockCollection("feedback")

mongodb_mod.get_users_collection = lambda: mock_users
mongodb_mod.get_collections_collection = lambda: mock_collections
mongodb_mod.get_usage_history_collection = lambda: mock_usage_history
mongodb_mod.get_feedback_collection = lambda: mock_feedback

# ---------------------------------------------------------------------------
# Import aplikasi FastAPI
# ---------------------------------------------------------------------------

# Pastikan working directory adalah root project (stackmatch-backend/) saat
# menjalankan pytest agar import relatif berfungsi.
from main import app  # noqa: E402

# Patch module local references yang di-import via 'from ... import ...'
import app.utils.auth as auth_mod
import app.routes.collections as col_route
import app.routes.recommend as rec_route
import app.routes.analytics as aly_route
import app.services.model_service as model_serv

auth_mod.get_users_collection = lambda: mock_users
col_route.get_collections_collection = lambda: mock_collections
rec_route.get_usage_history_collection = lambda: mock_usage_history
aly_route.get_feedback_collection = lambda: mock_feedback
aly_route.get_usage_history_collection = lambda: mock_usage_history
model_serv.get_feedback_collection = lambda: mock_feedback
model_serv.get_usage_history_collection = lambda: mock_usage_history


# ---------------------------------------------------------------------------
# Konfigurasi pytest-asyncio
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Base URL & helper konstanta
# ---------------------------------------------------------------------------
BASE = "/api/v1"
AUTH = f"{BASE}/auth"
DATA = f"{BASE}/data"
RECOMMEND = f"{BASE}/recommend"
SEARCH = f"{BASE}/recommend/search"
ANALYTICS = f"{BASE}/analytics"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def reset_mock_db():
    mock_users.documents.clear()
    mock_collections.documents.clear()
    mock_usage_history.documents.clear()
    mock_feedback.documents.clear()


@pytest_asyncio.fixture
async def client():
    """AsyncClient yang terhubung langsung ke app ASGI (tanpa server nyata)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


def _unique_email() -> str:
    """Hasilkan email unik agar test tidak bentrok jika dijalankan berulang."""
    return f"testuser_{uuid.uuid4().hex[:8]}@example.com"


def _unique_username() -> str:
    return f"testuser_{uuid.uuid4().hex[:6]}"


async def _register_and_login(client: AsyncClient) -> dict[str, Any]:
    """
    Helper: daftarkan user baru lalu login, kembalikan payload login lengkap.
    Digunakan oleh test yang membutuhkan token autentikasi.
    """
    email = _unique_email()
    username = _unique_username()
    password = "Password123"

    # Register
    reg = await client.post(
        f"{AUTH}/register",
        json={"username": username, "email": email, "password": password},
    )
    assert reg.status_code == 201, f"Register gagal: {reg.text}"

    # Clear auth rate limiter to avoid 429
    from app.routes.auth import _rate_store
    _rate_store.clear()

    # Login
    login = await client.post(
        f"{AUTH}/login",
        json={"email": email, "password": password},
    )
    assert login.status_code == 200, f"Login gagal: {login.text}"
    return login.json()


# ===========================================================================
# 1. test_register_success
# ===========================================================================

async def test_register_success(client: AsyncClient):
    """
    POST /auth/register dengan data valid harus mengembalikan:
      - HTTP 201
      - JSON dengan key 'user_id' yang tidak kosong
    """
    payload = {
        "username": _unique_username(),
        "email": _unique_email(),
        "password": "Secure123",
    }
    response = await client.post(f"{AUTH}/register", json=payload)

    assert response.status_code == 201, response.text
    data = response.json()
    assert "user_id" in data, f"'user_id' tidak ada di response: {data}"
    assert data["user_id"], "user_id tidak boleh kosong"


# ===========================================================================
# 2. test_register_duplicate_email
# ===========================================================================

async def test_register_duplicate_email(client: AsyncClient):
    """
    Dua kali registrasi dengan email yang sama harus mengembalikan HTTP 409.
    """
    email = _unique_email()
    payload = {
        "username": _unique_username(),
        "email": email,
        "password": "Secure123",
    }

    # Registrasi pertama (harus sukses)
    r1 = await client.post(f"{AUTH}/register", json=payload)
    assert r1.status_code == 201, f"Registrasi pertama gagal: {r1.text}"

    # Registrasi kedua dengan email sama (harus 409)
    payload2 = {**payload, "username": _unique_username()}  # username berbeda
    r2 = await client.post(f"{AUTH}/register", json=payload2)
    assert r2.status_code == 409, (
        f"Harus 409 untuk email duplikat, dapat: {r2.status_code} – {r2.text}"
    )


# ===========================================================================
# 3. test_login_success
# ===========================================================================

async def test_login_success(client: AsyncClient):
    """
    POST /auth/login dengan kredensial valid harus mengembalikan:
      - HTTP 200
      - 'access_token' dan 'refresh_token' yang tidak kosong
    """
    # Siapkan user
    email = _unique_email()
    username = _unique_username()
    reg = await client.post(
        f"{AUTH}/register",
        json={"username": username, "email": email, "password": "Pass1234"},
    )
    assert reg.status_code == 201

    # Login
    response = await client.post(
        f"{AUTH}/login",
        json={"email": email, "password": "Pass1234"},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert "access_token" in data and data["access_token"], \
        "'access_token' tidak ada atau kosong"
    assert "refresh_token" in data and data["refresh_token"], \
        "'refresh_token' tidak ada atau kosong"

async def test_translate_to_english_uses_local_fallback_for_indonesian_query():
    from app.services.model_service import _translate_to_english

    translated = _translate_to_english("koneksi database")
    assert translated == "connection database"


async def test_local_translate_replaces_indonesian_programming_terms():
    from app.services.model_service import _local_translate

    translated = _local_translate("cara koneksi database")
    assert translated == "how to connection database"

# ===========================================================================
# 4. test_login_wrong_password
# ===========================================================================

async def test_login_wrong_password(client: AsyncClient):
    """
    POST /auth/login dengan password salah harus mengembalikan HTTP 401.
    """
    email = _unique_email()
    await client.post(
        f"{AUTH}/register",
        json={
            "username": _unique_username(),
            "email": email,
            "password": "Correct123",
        },
    )

    response = await client.post(
        f"{AUTH}/login",
        json={"email": email, "password": "WrongPassword999"},
    )
    assert response.status_code == 401, (
        f"Harus 401 untuk password salah, dapat: {response.status_code}"
    )


# ===========================================================================
# 5. test_refresh_token
# ===========================================================================

async def test_refresh_token(client: AsyncClient):
    """
    POST /auth/refresh dengan refresh_token valid harus mengembalikan:
      - HTTP 200
      - 'access_token' baru yang tidak kosong
    """
    login_data = await _register_and_login(client)
    refresh_token = login_data["refresh_token"]
    old_access_token = login_data["access_token"]

    response = await client.post(
        f"{AUTH}/refresh",
        json={"refresh_token": refresh_token},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert "access_token" in data and data["access_token"], \
        "'access_token' tidak ada atau kosong di response refresh"
    # Token baru seharusnya berbeda dari yang lama (jti berbeda)
    assert data["access_token"] != old_access_token, \
        "access_token baru harus berbeda dari yang lama"


# ===========================================================================
# 6. test_logout_then_use_token
# ===========================================================================

async def test_logout_then_use_token(client: AsyncClient):
    """
    Setelah POST /auth/logout dengan token yang valid:
      - Logout harus sukses (200)
      - Request GET /auth/me dengan token lama harus mengembalikan 401
    """
    login_data = await _register_and_login(client)
    access_token = login_data["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    # Logout
    logout_resp = await client.post(f"{AUTH}/logout", headers=headers)
    assert logout_resp.status_code == 200, f"Logout gagal: {logout_resp.text}"

    # Coba akses endpoint yang dilindungi dengan token lama
    me_resp = await client.get(f"{AUTH}/me", headers=headers)
    assert me_resp.status_code == 401, (
        f"Harus 401 setelah logout, dapat: {me_resp.status_code} – {me_resp.text}"
    )


# ===========================================================================
# 7. test_recommend_valid_query
# ===========================================================================

async def test_recommend_valid_query(client: AsyncClient):
    """
    POST /recommend dengan query valid harus mengembalikan:
      - HTTP 200
      - 'results' berupa list (bisa kosong jika model belum dimuat)
      - Setiap item dalam results punya field: id, question, score_fusion,
        score_tfidf, score_sbert, tag
    """
    response = await client.post(
        RECOMMEND,
        json={"query": "how to sort a list in python", "top_n": 5},
    )

    # Jika model belum dimuat, endpoint bisa kembalikan 200 dengan results []
    # atau 500 (RuntimeError). Kedua kondisi ini diterima dalam test environment.
    if response.status_code == 500:
        pytest.skip("Model ML belum dimuat – /recommend tidak bisa diuji penuh")

    assert response.status_code == 200, response.text
    data = response.json()
    assert "results" in data, f"'results' tidak ada di response: {data}"
    assert isinstance(data["results"], list)

    # Jika ada hasil, validasi structure setiap item
    required_score_fields = {"id", "question", "score_fusion", "score_tfidf", "score_sbert", "tag"}
    for item in data["results"]:
        missing = required_score_fields - item.keys()
        assert not missing, f"Fields kurang di item hasil: {missing} — item: {item}"


# ===========================================================================
# 8. test_recommend_empty_query
# ===========================================================================

async def test_recommend_empty_query(client: AsyncClient):
    """
    POST /recommend dengan query kosong harus mengembalikan HTTP 422
    (Pydantic validation error) atau 400.
    """
    response = await client.post(
        RECOMMEND,
        json={"query": "   ", "top_n": 5},
    )
    # Pydantic v2 akan raise 422 untuk field_validator error
    assert response.status_code in (400, 422), (
        f"Harus 400 atau 422 untuk query kosong, dapat: {response.status_code}"
    )


# ===========================================================================
# 9. test_search_invalid_tag
# ===========================================================================

async def test_search_invalid_tag(client: AsyncClient):
    """
    POST /recommend/search dengan tag yang tidak ada di valid_tags harus
    mengembalikan HTTP 400 beserta daftar tag yang valid.
    """
    response = await client.post(
        SEARCH,
        json={
            "query": "how to handle errors",
            "tag": "invalidtagxyz123",
            "top_n": 5,
        },
    )

    assert response.status_code == 400, (
        f"Harus 400 untuk tag tidak valid, dapat: {response.status_code} – {response.text}"
    )
    data = response.json()
    # Pastikan response mengandung informasi error yang membantu
    detail = data.get("detail") or data.get("error") or data
    assert detail, "Response 400 harus mengandung detail error"


# ===========================================================================
# 10. test_questions_pagination
# ===========================================================================

async def test_questions_pagination(client: AsyncClient):
    """
    GET /data/questions?limit=5&offset=0 harus mengembalikan:
      - HTTP 200 (atau 503 jika dataset belum dimuat)
      - 'total' ada di response
      - Jumlah 'questions' ≤ limit yang diminta
    """
    response = await client.get(f"{DATA}/questions", params={"limit": 5, "offset": 0})

    if response.status_code == 503:
        pytest.skip("Dataset belum dimuat – /data/questions tidak bisa diuji")

    assert response.status_code == 200, response.text
    data = response.json()

    assert "total" in data, f"'total' tidak ada di response: {data}"
    assert "questions" in data, f"'questions' tidak ada di response: {data}"
    assert isinstance(data["questions"], list)
    assert len(data["questions"]) <= 5, (
        f"Jumlah item melebihi limit=5: {len(data['questions'])}"
    )

    # Test offset: halaman kedua berbeda dari halaman pertama
    response2 = await client.get(
        f"{DATA}/questions", params={"limit": 5, "offset": 5}
    )
    if response2.status_code == 200:
        data2 = response2.json()
        if data["questions"] and data2["questions"]:
            assert data["questions"][0] != data2["questions"][0], \
                "Item pertama di offset=0 dan offset=5 seharusnya berbeda"


# ===========================================================================
# 11. test_random_query_string
# ===========================================================================

async def test_random_query_string(client: AsyncClient):
    """
    GET /data/random?n=3&tag=python harus mengembalikan:
      - HTTP 200 (atau 503 jika dataset belum dimuat)
      - 'questions' berupa list dengan ≤ 3 item
    """
    response = await client.get(
        f"{DATA}/random",
        params={"n": 3, "tag": "python"},
    )

    if response.status_code == 503:
        pytest.skip("Dataset belum dimuat – /data/random tidak bisa diuji")

    assert response.status_code == 200, response.text
    data = response.json()

    assert "questions" in data, f"'questions' tidak ada di response: {data}"
    assert isinstance(data["questions"], list)
    assert len(data["questions"]) <= 3, (
        f"Jumlah item melebihi n=3: {len(data['questions'])}"
    )

    # Tanpa filter tag
    response2 = await client.get(f"{DATA}/random", params={"n": 5})
    if response2.status_code == 200:
        data2 = response2.json()
        assert "questions" in data2
        assert len(data2["questions"]) <= 5


# ===========================================================================
# 12. test_feedback_post_then_get
# ===========================================================================

async def test_feedback_post_then_get(client: AsyncClient):
    """
    Setelah POST /analytics/feedback dengan data valid:
      - Response harus 201
      - GET /analytics/feedback?question_id=<id> harus mengembalikan has_rated: true

    Catatan: question_id 1 diasumsikan ada di dataset atau dilewati validasinya
    (dataset belum dimuat → _question_exists() kembalikan True).
    """
    login_data = await _register_and_login(client)
    access_token = login_data["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    question_id = 1  # ID yang diasumsikan ada / validasi dilewati jika dataset kosong

    # POST feedback
    post_resp = await client.post(
        f"{ANALYTICS}/feedback",
        json={
            "question_id": question_id,
            "query": "how to sort a list",
            "rating": 4,
            "comment": "Sangat relevan!",
        },
        headers=headers,
    )

    if post_resp.status_code == 400:
        # question_id tidak ditemukan di dataset yang sudah dimuat
        pytest.skip(
            f"question_id={question_id} tidak ditemukan di dataset – "
            "feedback test dilewati"
        )

    assert post_resp.status_code == 201, (
        f"POST feedback harus 201, dapat: {post_resp.status_code} – {post_resp.text}"
    )
    post_data = post_resp.json()
    assert "feedback_id" in post_data, f"'feedback_id' tidak ada: {post_data}"

    # GET feedback untuk memverifikasi has_rated: true
    get_resp = await client.get(
        f"{ANALYTICS}/feedback",
        params={"question_id": question_id},
        headers=headers,
    )
    assert get_resp.status_code == 200, (
        f"GET feedback harus 200, dapat: {get_resp.status_code} – {get_resp.text}"
    )
    get_data = get_resp.json()
    assert get_data.get("has_rated") is True, (
        f"has_rated harus True setelah POST feedback, dapat: {get_data}"
    )
    assert get_data.get("rating") == 4, (
        f"Rating harus 4, dapat: {get_data.get('rating')}"
    )


# ===========================================================================
# 13. test_health
# ===========================================================================

async def test_health(client: AsyncClient):
    """
    GET /data/health harus mengembalikan:
      - HTTP 200
      - 'model_loaded' ada di response (True jika model dimuat, False jika tidak)
      - 'status' == 'ok'
      - 'version' ada
    """
    response = await client.get(f"{DATA}/health")

    assert response.status_code == 200, response.text
    data = response.json()

    assert "model_loaded" in data, f"'model_loaded' tidak ada di response: {data}"
    assert "status" in data, f"'status' tidak ada di response: {data}"
    assert data["status"] == "ok", f"status harus 'ok', dapat: {data['status']}"
    assert "version" in data, f"'version' tidak ada di response: {data}"

    # model_loaded bisa True atau False tergantung environment
    assert isinstance(data["model_loaded"], bool), \
        f"'model_loaded' harus boolean, dapat: {type(data['model_loaded'])}"


# ===========================================================================
# Bonus: test root health endpoint
# ===========================================================================

async def test_root_health(client: AsyncClient):
    """
    GET /api/v1/health (root health check di main.py) harus mengembalikan 200.
    """
    response = await client.get(f"{BASE}/health")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data.get("status") == "ok"
    assert data.get("service") == "StackMatch API"


# ===========================================================================
# Bonus: test me endpoint
# ===========================================================================

async def test_me_authenticated(client: AsyncClient):
    """
    GET /auth/me dengan token valid harus mengembalikan profil user.
    """
    login_data = await _register_and_login(client)
    access_token = login_data["access_token"]

    response = await client.get(
        f"{AUTH}/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert "id" in data
    assert "username" in data
    assert "email" in data


async def test_me_unauthenticated(client: AsyncClient):
    """
    GET /auth/me tanpa header Authorization harus mengembalikan 403.
    """
    response = await client.get(f"{AUTH}/me")
    assert response.status_code in (401, 403), (
        f"Harus 401 atau 403 tanpa token, dapat: {response.status_code}"
    )


# ===========================================================================
# 14. test_feedback_batch
# ===========================================================================

async def test_feedback_batch_success(client: AsyncClient):
    """
    POST feedback untuk beberapa ID, lalu panggil GET /analytics/feedback/batch.
    """
    login_data = await _register_and_login(client)
    access_token = login_data["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    # Kirim feedback 1
    post1 = await client.post(
        f"{ANALYTICS}/feedback",
        json={
            "question_id": 1,
            "query": "test query 1",
            "rating": 4,
            "comment": "Nice",
        },
        headers=headers,
    )
    if post1.status_code == 400:
        pytest.skip("Dataset loaded dan ID 1 tidak valid di dataset.")

    # Kirim feedback 2
    await client.post(
        f"{ANALYTICS}/feedback",
        json={
            "question_id": 2,
            "query": "test query 2",
            "rating": 5,
        },
        headers=headers,
    )

    # Batch GET
    response = await client.get(
        f"{ANALYTICS}/feedback/batch",
        params={"question_ids": "1,2,3"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert "ratings" in data
    ratings = data["ratings"]

    assert ratings["1"]["has_rated"] is True
    assert ratings["1"]["rating"] == 4
    assert ratings["2"]["has_rated"] is True
    assert ratings["2"]["rating"] == 5
    assert ratings["3"]["has_rated"] is False
    assert "rating" not in ratings["3"]  # excluded because of exclude_none


async def test_feedback_batch_limits(client: AsyncClient):
    """
    Validasi batas parameter pada GET /analytics/feedback/batch.
    """
    login_data = await _register_and_login(client)
    access_token = login_data["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    # Lebih dari 50 ID
    ids_too_many = ",".join(str(i) for i in range(51))
    r_limit = await client.get(
        f"{ANALYTICS}/feedback/batch",
        params={"question_ids": ids_too_many},
        headers=headers,
    )
    assert r_limit.status_code == 400

    # ID tidak valid
    r_invalid = await client.get(
        f"{ANALYTICS}/feedback/batch",
        params={"question_ids": "1,abc,3"},
        headers=headers,
    )
    assert r_invalid.status_code == 400

    # Parameter kosong
    r_empty = await client.get(
        f"{ANALYTICS}/feedback/batch",
        params={"question_ids": ""},
        headers=headers,
    )
    assert r_empty.status_code == 400


async def test_feedback_batch_unauthenticated(client: AsyncClient):
    """
    GET /analytics/feedback/batch tanpa token harus mengembalikan 401/403.
    """
    response = await client.get(
        f"{ANALYTICS}/feedback/batch",
        params={"question_ids": "1,2"},
    )
    assert response.status_code in (401, 403)


# ===========================================================================
# 15. test_personalization
# ===========================================================================

async def test_get_user_preference_vector_behavior():
    """
    Test get_user_preference_vector for cold start threshold and correct calculation.
    """
    from app.services.model_service import get_user_preference_vector, MODEL_STORE
    import pandas as pd

    test_user = "test_personalization_user_123"

    # Pastikan model_store punya dataset mock untuk keperluan test ini jika kosong
    orig_dataset = MODEL_STORE.get("dataset")
    mock_df = pd.DataFrame({
        "Title": ["Java Question", "Python Question", "Javascript Question"],
        "Tags": ["java", "python", "javascript"],
        "AnswerBody": ["body1", "body2", "body3"],
        "AnswerScore": [10.0, 20.0, 30.0]
    })
    # Set index agar representatif
    mock_df.index = [0, 1, 2]
    MODEL_STORE["dataset"] = mock_df

    try:
        # A. Uji Cold Start (< 3 interaksi)
        # 1 feedback + 1 history = 2 interactions (< 3)
        mock_feedback.documents = [
            {"user_id": test_user, "question_id": 0, "rating": 5, "timestamp": "2026-05-24T09:45:18.579447+00:00"}
        ]
        mock_usage_history.documents = [
            {"user_id": test_user, "tag": "python", "timestamp": "2026-05-24T07:26:44.675280+00:00"}
        ]

        pref = await get_user_preference_vector(test_user)
        assert pref is None, "Harus returns None untuk interaksi < 3 (cold start)"

        # B. Uji >= 3 interaksi
        # Tambah 1 usage history -> total 3 interaksi
        mock_usage_history.documents = [
            {"user_id": test_user, "tag": "python", "timestamp": "2026-05-24T07:26:44.675280+00:00"},
            {"user_id": test_user, "tag": "java", "timestamp": "2026-05-24T07:42:27.183013+00:00"}
        ]

        pref = await get_user_preference_vector(test_user)
        assert pref is not None, "Harus menghasilkan pref vector ketika interaksi >= 3"
        assert "tag_weights" in pref
        assert "preferred_answer_score_min" in pref

        # Check specific values:
        # rated question 0 ("java") rating 5. -> feedback_factor = 5.0 / 3.0 = 1.67.
        # searched tags: "python" (count 1), "java" (count 1). Total searches = 2.
        # frequency_factor for java = 1.0 + (1/2)*0.5 = 1.25.
        # weight for java = 1.67 * 1.25 = 2.08 -> capped at 2.00
        # answer_score of highly rated question (rating 5 >= 4) is 10.0.
        assert pref["tag_weights"].get("java") == 2.0
        assert pref["preferred_answer_score_min"] == 10.0

    finally:
        if orig_dataset is not None:
            MODEL_STORE["dataset"] = orig_dataset
        else:
            MODEL_STORE.pop("dataset", None)


async def test_personalized_hybrid_search():
    """
    Test hybrid_search with user_preference tag weight multiplier.
    """
    from app.services.model_service import hybrid_search, MODEL_STORE
    import pandas as pd

    # Backup original store
    orig_store = MODEL_STORE.copy()

    # Create mock dataset
    mock_df = pd.DataFrame({
        "Title": ["Java Question", "Python Question"],
        "Tags": ["java", "python"],
        "AnswerBody": ["body1", "body2"],
        "AnswerScore": [10.0, 10.0]
    })
    mock_df.index = [0, 1]

    MODEL_STORE["dataset"] = mock_df
    MODEL_STORE["config"] = {
        "alpha": 0.0,  # TF-IDF weight
        "beta": 0.0,   # SBERT weight
        "gamma": 1.0,  # answer score weight
    }
    MODEL_STORE["vectorizer"] = None
    MODEL_STORE["tfidf_matrix"] = None
    MODEL_STORE["faiss_index"] = None
    MODEL_STORE["sbert"] = None

    try:
        # Tanpa preferensi, skor sama
        res_normal = await hybrid_search(query="test", top_n=2)
        assert len(res_normal) == 2
        assert res_normal[0]["score"] == 1.0
        assert res_normal[1]["score"] == 1.0

        # Dengan preferensi: Python di-boost, Java diturunkan
        user_pref = {
            "tag_weights": {"python": 1.5, "java": 0.5},
            "preferred_answer_score_min": 0.0
        }
        res_personalized = await hybrid_search(query="test", top_n=2, user_preference=user_pref)
        assert len(res_personalized) == 2
        assert res_personalized[0]["tag"] == "python"
        assert res_personalized[0]["score"] == 1.5
        assert res_personalized[1]["tag"] == "java"
        assert res_personalized[1]["score"] == 0.5

    finally:
        # Restore store
        MODEL_STORE.clear()
        MODEL_STORE.update(orig_store)


# ===========================================================================
# 16. test_ai_answer_synthesis
# ===========================================================================

async def test_synthesize_unauthenticated(client: AsyncClient):
    """
    POST /recommend/synthesize tanpa token harus mengembalikan 401/403.
    """
    payload = {
        "query": "how to sort",
        "results": [
            {"question": "Q1", "answer_full": "A1", "score_fusion": 0.9, "tag": "python"}
        ],
        "language": "id"
    }
    response = await client.post(f"{RECOMMEND}/synthesize", json=payload)
    assert response.status_code in (401, 403)


async def test_synthesize_missing_api_key(client: AsyncClient):
    """
    POST /recommend/synthesize tanpa GEMINI_API_KEY harus mengembalikan 503.
    """
    login_data = await _register_and_login(client)
    access_token = login_data["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    payload = {
        "query": "how to sort",
        "results": [
            {"question": "Q1", "answer_full": "A1", "score_fusion": 0.9, "tag": "python"}
        ],
        "language": "id"
    }

    import os
    from unittest.mock import patch
    with patch.dict(os.environ, {}, clear=True):
        # We cleared the env, so GEMINI_API_KEY is not present
        response = await client.post(f"{RECOMMEND}/synthesize", json=payload, headers=headers)
        assert response.status_code == 503
        assert "GEMINI_API_KEY" in response.json()["error"]


async def test_synthesize_rate_limiting(client: AsyncClient):
    """
    POST /recommend/synthesize melebihi 10 kali dalam satu menit harus mengembalikan 429.
    """
    login_data = await _register_and_login(client)
    access_token = login_data["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    payload = {
        "query": "how to sort",
        "results": [
            {"question": "Q1", "answer_full": "A1", "score_fusion": 0.9, "tag": "python"}
        ],
        "language": "id"
    }

    import os
    import httpx
    from unittest.mock import patch

    class MockResponse:
        def __init__(self):
            self.status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {
                "content": [
                    {"text": '{"synthesis": "Summary...", "key_points": ["Point A"]}'}
                ]
            }

    real_post = httpx.AsyncClient.post

    async def mock_post(self, url, *args, **kwargs):
        if "generativelanguage.googleapis.com" in str(url) or "gemini.googleapis.com" in str(url):
            return MockResponse()
        return await real_post(self, url, *args, **kwargs)

    # Clear rate limit store for this user first
    from app.routes.recommend import _synth_rate_store
    user_id = login_data["user"]["id"]
    if user_id in _synth_rate_store:
        _synth_rate_store[user_id] = []

    with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_key"}), patch("httpx.AsyncClient.post", mock_post):
        # Send 10 successful requests with unique queries to avoid cache hits
        for i in range(10):
            payload["query"] = f"how to sort {i}"
            r = await client.post(f"{RECOMMEND}/synthesize", json=payload, headers=headers)
            assert r.status_code == 200, f"Request {i+1} failed: {r.text}"
            
        # 11th request must return 429
        payload["query"] = "how to sort 10"
        r_11 = await client.post(f"{RECOMMEND}/synthesize", json=payload, headers=headers)
        assert r_11.status_code == 429
        assert "terlampaui" in r_11.json()["error"]


async def test_synthesize_success(client: AsyncClient):
    """
    POST /recommend/synthesize dengan parameter lengkap mengembalikan status 200 dan respon rangkuman.
    """
    login_data = await _register_and_login(client)
    access_token = login_data["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    payload = {
        "query": "how to sort list",
        "results": [
            {"question": "Q1", "answer_full": "A1", "score_fusion": 0.8, "tag": "python"},
            {"question": "Q2", "answer_full": "A2", "score_fusion": 0.6, "tag": "python"}
        ],
        "language": "en"
    }

    import os
    import httpx
    from unittest.mock import patch
    
    class MockResponse:
        def __init__(self):
            self.status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {
                "content": [
                    {"text": '{"synthesis": "This is a synthesized text.", "key_points": ["Point 1", "Point 2"]}'}
                ]
            }

    real_post = httpx.AsyncClient.post

    async def mock_post(self, url, *args, **kwargs):
        if "generativelanguage.googleapis.com" in str(url) or "gemini.googleapis.com" in str(url):
            return MockResponse()
        return await real_post(self, url, *args, **kwargs)

    # Clear rate limit store for this user to avoid 429 from the previous test
    from app.routes.recommend import _synth_rate_store
    user_id = login_data["user"]["id"]
    if user_id in _synth_rate_store:
        _synth_rate_store[user_id] = []

    with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_key"}), patch("httpx.AsyncClient.post", mock_post):
        response = await client.post(f"{RECOMMEND}/synthesize", json=payload, headers=headers)
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["synthesis"] == "This is a synthesized text."
        assert data["key_points"] == ["Point 1", "Point 2"]
        # confidence is mean(0.8, 0.6) = 0.7
        assert data["confidence"] == 0.7
        assert data["sources_used"] == 2


async def test_synthesize_caching_behavior(client: AsyncClient):
    """
    POST /recommend/synthesize should cache the result in MongoDB.
    Subsequent requests with the same query should bypass the Gemini API and rate limiter,
    returning the cached result. Also GET /analytics/usage must return the synthesis details.
    """
    login_data = await _register_and_login(client)
    access_token = login_data["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    payload = {
        "query": "how to sort list cached",
        "results": [
            {"question": "Q1", "answer_full": "A1", "score_fusion": 0.8, "tag": "python"}
        ],
        "language": "en"
    }

    import os
    import httpx
    from unittest.mock import patch
    
    class MockResponseFirst:
        def __init__(self):
            self.status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {
                "content": [
                    {"text": '{"synthesis": "First time synthesis", "key_points": ["FP1"]}'}
                ]
            }

    class MockResponseSecond:
        def __init__(self):
            self.status_code = 500
        def raise_for_status(self):
            raise httpx.HTTPStatusError("Should not call API on cache hit", request=None, response=self)

    real_post = httpx.AsyncClient.post
    call_count = 0

    async def mock_post(self, url, *args, **kwargs):
        nonlocal call_count
        if "generativelanguage.googleapis.com" in str(url) or "gemini.googleapis.com" in str(url):
            call_count += 1
            if call_count == 1:
                return MockResponseFirst()
            else:
                return MockResponseSecond()
        return await real_post(self, url, *args, **kwargs)

    # Clear rate limit store for this user
    from app.routes.recommend import _synth_rate_store
    user_id = login_data["user"]["id"]
    if user_id in _synth_rate_store:
        _synth_rate_store[user_id] = []

    with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_key"}), patch("httpx.AsyncClient.post", mock_post):
        # 1. First call (should hit Gemini API and save to MongoDB)
        response1 = await client.post(f"{RECOMMEND}/synthesize", json=payload, headers=headers)
        assert response1.status_code == 200, response1.text
        assert response1.json()["synthesis"] == "First time synthesis"
        assert call_count == 1

        # 2. Second call (should hit cache in MongoDB and bypass Gemini API completely)
        # Even if we change GEMINI_API_KEY to empty (which normally raises 503), it should bypass it!
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            response2 = await client.post(f"{RECOMMEND}/synthesize", json=payload, headers=headers)
            assert response2.status_code == 200, response2.text
            assert response2.json()["synthesis"] == "First time synthesis"
            # call_count should STILL be 1 because Gemini API was never called!
            assert call_count == 1

        # 3. GET /analytics/usage should return the synthesis in history
        history_resp = await client.get(f"{ANALYTICS}/usage", headers=headers)
        assert history_resp.status_code == 200, history_resp.text
        history_data = history_resp.json()["history"]
        
        # Verify the most recent history item contains the cached synthesis
        matched_item = None
        for item in history_data:
            if item["query"] == "how to sort list cached":
                matched_item = item
                break
        
        assert matched_item is not None
        assert matched_item["synthesis"] == "First time synthesis"
        assert matched_item["key_points"] == ["FP1"]


# ===========================================================================
# 17. test_smart_query_expansion
# ===========================================================================

async def test_generate_query_suggestions_behavior():
    """
    Test generate_query_suggestions helper logic.
    """
    from app.services.model_service import generate_query_suggestions, MODEL_STORE
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd

    orig_store = MODEL_STORE.copy()

    # Mock TfidfVectorizer
    vectorizer = TfidfVectorizer()
    corpus = ["how to sort a list in python", "python list sorting tutorial", "debugging python list errors"]
    vectorizer.fit(corpus)

    MODEL_STORE["vectorizer"] = vectorizer

    mock_results = [
        {"question": "Python list sorting", "answer_full": "Use sorted() function", "tag": "python", "score_fusion": 0.9},
        {"question": "Javascript sorting", "answer_full": "Use Array.prototype.sort", "tag": "javascript", "score_fusion": 0.8},
        {"question": "Java sorting", "answer_full": "Use Collections.sort", "tag": "java", "score_fusion": 0.7}
    ]

    try:
        suggestions = generate_query_suggestions(
            query="sorting",
            results=mock_results,
            dataset=None,
            filtered_tag="javascript"  # Exclude javascript from related_tags
        )
        assert "related_queries" in suggestions
        assert "related_tags" in suggestions
        assert len(suggestions["related_queries"]) == 3
        # Excluded tag javascript should not be in related_tags
        assert "javascript" not in suggestions["related_tags"]
        # Tags present in results should be in related_tags: python, java
        assert "python" in suggestions["related_tags"]
        assert "java" in suggestions["related_tags"]

    finally:
        MODEL_STORE.clear()
        MODEL_STORE.update(orig_store)


async def test_recommend_endpoint_returns_suggestions(client: AsyncClient):
    """
    Test that /recommend returns suggestions when mocked results are returned.
    """
    from app.services.model_service import MODEL_STORE
    import pandas as pd

    orig_store = MODEL_STORE.copy()

    # Mock dataset and empty ML artifacts
    mock_df = pd.DataFrame({
        "Title": ["Java Question", "Python Question"],
        "Tags": ["java", "python"],
        "AnswerBody": ["body1", "body2"],
        "AnswerScore": [10.0, 10.0]
    })
    mock_df.index = [0, 1]

    MODEL_STORE["dataset"] = mock_df
    MODEL_STORE["config"] = {
        "alpha": 0.0,
        "beta": 0.0,
        "gamma": 1.0,
    }
    MODEL_STORE["vectorizer"] = None
    MODEL_STORE["tfidf_matrix"] = None
    MODEL_STORE["faiss_index"] = None
    MODEL_STORE["sbert"] = None

    try:
        response = await client.post(RECOMMEND, json={"query": "test", "top_n": 2})
        assert response.status_code == 200, response.text
        data = response.json()
        assert "results" in data
        assert "suggestions" in data
        suggestions = data["suggestions"]
        assert "related_queries" in suggestions
        assert "related_tags" in suggestions
        assert len(suggestions["related_queries"]) == 3
        # tags of mock dataset: java, python
        assert "java" in suggestions["related_tags"]
        assert "python" in suggestions["related_tags"]

    finally:
        MODEL_STORE.clear()
        MODEL_STORE.update(orig_store)


# ===========================================================================
# 18. Saved Collections Tests
# ===========================================================================

@pytest.fixture
def mock_collections_storage(tmp_path):
    import app.routes.collections as col_mod
    from unittest.mock import patch
    temp_file = tmp_path / "collections.json"
    with patch.object(col_mod, "_COLLECTIONS_FILE", temp_file):
        yield temp_file


async def test_collections_lifecycle(client: AsyncClient, mock_collections_storage):
    """
    Uji lifecycle Saved Collections:
    - POST /collections -> sukses 201
    - POST /collections (nama duplikat) -> error 409
    - POST /collections/{id}/items -> sukses 201
    - POST /collections/{id}/items (duplikat) -> error 409
    - GET /collections -> list metadata, item_count=1
    - GET /collections/{id} -> detail, berisi 1 item
    - DELETE /collections/{id}/items/{question_id} -> sukses 200
    - GET /collections/{id} -> detail, items kosong
    - DELETE /collections/{id} -> sukses 200
    - GET /collections/{id} -> error 404
    """
    login_data = await _register_and_login(client)
    access_token = login_data["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    # 1. Create collection
    create_resp = await client.post(
        f"{BASE}/collections",
        json={"name": "My Test Collection"},
        headers=headers
    )
    assert create_resp.status_code == 201, create_resp.text
    col_data = create_resp.json()
    assert "id" in col_data
    assert col_data["name"] == "My Test Collection"
    assert col_data["item_count"] == 0
    col_id = col_data["id"]

    # 2. Duplicate collection name check
    dup_resp = await client.post(
        f"{BASE}/collections",
        json={"name": "  My Test Collection  "},
        headers=headers
    )
    assert dup_resp.status_code == 409

    # 3. Add item to collection
    item_payload = {
        "question_id": "123",
        "question": "How to do X?",
        "answer_preview": "Do Y",
        "tag": "python",
        "score_fusion": 0.95,
        "note": "Useful tip"
    }
    item_resp = await client.post(
        f"{BASE}/collections/{col_id}/items",
        json=item_payload,
        headers=headers
    )
    assert item_resp.status_code == 201, item_resp.text
    item_data = item_resp.json()
    assert item_data["question_id"] == "123"
    assert item_data["note"] == "Useful tip"
    assert "saved_at" in item_data

    # 4. Duplicate item check
    dup_item_resp = await client.post(
        f"{BASE}/collections/{col_id}/items",
        json=item_payload,
        headers=headers
    )
    assert dup_item_resp.status_code == 409

    # 5. List collections
    list_resp = await client.get(f"{BASE}/collections", headers=headers)
    assert list_resp.status_code == 200
    collections = list_resp.json()
    assert len(collections) == 1
    assert collections[0]["id"] == col_id
    assert collections[0]["item_count"] == 1

    # 6. Get collection details
    detail_resp = await client.get(f"{BASE}/collections/{col_id}", headers=headers)
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["name"] == "My Test Collection"
    assert len(detail["items"]) == 1
    assert detail["items"][0]["question_id"] == "123"

    # 7. Remove item from collection
    remove_resp = await client.delete(
        f"{BASE}/collections/{col_id}/items/123",
        headers=headers
    )
    assert remove_resp.status_code == 200

    # 8. Verify item is removed
    detail_resp2 = await client.get(f"{BASE}/collections/{col_id}", headers=headers)
    assert detail_resp2.status_code == 200
    assert len(detail_resp2.json()["items"]) == 0

    # 9. Delete collection
    del_resp = await client.delete(f"{BASE}/collections/{col_id}", headers=headers)
    assert del_resp.status_code == 200

    # 10. Verify collection is deleted
    detail_resp3 = await client.get(f"{BASE}/collections/{col_id}", headers=headers)
    assert detail_resp3.status_code == 404


async def test_collections_limits(client: AsyncClient, mock_collections_storage):
    """
    Uji batasan limit Saved Collections:
    - Maksimal 20 koleksi per user.
    - Maksimal 100 item per koleksi.
    """
    login_data = await _register_and_login(client)
    access_token = login_data["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    # Create 20 collections
    col_ids = []
    for i in range(20):
        resp = await client.post(
            f"{BASE}/collections",
            json={"name": f"Collection {i}"},
            headers=headers
        )
        assert resp.status_code == 201
        col_ids.append(resp.json()["id"])

    # 21st collection should fail
    resp_21 = await client.post(
        f"{BASE}/collections",
        json={"name": "Collection 20"},
        headers=headers
    )
    assert resp_21.status_code == 400
    assert "Maksimal koleksi" in resp_21.json()["error"]

    # Target the first collection and add 100 items
    target_col_id = col_ids[0]
    for j in range(100):
        item_resp = await client.post(
            f"{BASE}/collections/{target_col_id}/items",
            json={
                "question_id": f"q_{j}",
                "question": f"Question {j}?",
                "answer_preview": f"Answer {j}",
                "tag": "python",
                "score_fusion": 0.9
            },
            headers=headers
        )
        assert item_resp.status_code == 201

    # 101st item should fail
    item_resp_101 = await client.post(
        f"{BASE}/collections/{target_col_id}/items",
        json={
            "question_id": "q_100",
            "question": "Question 100?",
            "answer_preview": "Answer 100",
            "tag": "python",
            "score_fusion": 0.9
        },
        headers=headers
    )
    assert item_resp_101.status_code == 400
    assert "Maksimal item" in item_resp_101.json()["error"]


async def test_collections_unauthenticated(client: AsyncClient, mock_collections_storage):
    """
    Uji request collections tanpa autentikasi (Authorization header) -> harus 401/403.
    """
    # GET /collections
    r1 = await client.get(f"{BASE}/collections")
    assert r1.status_code in (401, 403)

    # POST /collections
    r2 = await client.post(f"{BASE}/collections", json={"name": "No Auth"})
    assert r2.status_code in (401, 403)

    # DELETE /collections/123
    r3 = await client.delete(f"{BASE}/collections/123")
    assert r3.status_code in (401, 403)

    # POST /collections/123/items
    r4 = await client.post(
        f"{BASE}/collections/123/items",
        json={
            "question_id": "q1",
            "question": "Q?",
            "answer_preview": "A",
            "tag": "python",
            "score_fusion": 0.8
        }
    )
    assert r4.status_code in (401, 403)

    # DELETE /collections/123/items/q1
    r5 = await client.delete(f"{BASE}/collections/123/items/q1")
    assert r5.status_code in (401, 403)

    # GET /collections/123
    r6 = await client.get(f"{BASE}/collections/123")
    assert r6.status_code in (401, 403)


# ===========================================================================
# 19. Trending Dashboard Tests
# ===========================================================================

@pytest.fixture
def mock_usage_history_storage(tmp_path):
    import app.routes.analytics as analytics_mod
    from unittest.mock import patch
    temp_file = tmp_path / "usage_history.json"
    temp_file.write_text("[]", encoding="utf-8")
    with patch.object(analytics_mod, "_USAGE_FILE", temp_file):
        yield temp_file


async def test_trending_dashboard(client: AsyncClient, mock_usage_history_storage):
    """
    Uji endpoint /analytics/trending:
    - Dengan data kosong
    - Dengan beberapa data pencarian (today, past week, older)
    - Verifikasi perhitungan statistik & pengelompokan query
    """
    from datetime import datetime, timezone, timedelta
    import json

    # 1. Test empty state
    resp = await client.get(f"{BASE}/analytics/trending")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["trending_queries"] == []
    assert data["tag_distribution"] == []
    assert data["total_searches_today"] == 0
    assert data["total_searches_week"] == 0
    assert data["active_users_today"] == 0

    # 2. Populate usage history with mock records
    now = datetime.now(tz=timezone.utc)
    
    mock_history = [
        # Search today by user_1
        {
            "id": "1",
            "user_id": "user_1",
            "query": "konek database",
            "tag": "java",
            "timestamp": now.isoformat(),
            "result_count": 5
        },
        # Another search today by user_1 for same query
        {
            "id": "2",
            "user_id": "user_1",
            "query": "KONEK DATABASE",
            "tag": "php",
            "timestamp": (now - timedelta(minutes=10)).isoformat(),
            "result_count": 5
        },
        # Search today by user_2 for different query
        {
            "id": "3",
            "user_id": "user_2",
            "query": "how to sort a list",
            "tag": "python",
            "timestamp": (now - timedelta(hours=1)).isoformat(),
            "result_count": 3
        },
        # Search 3 days ago (this week)
        {
            "id": "4",
            "user_id": "user_3",
            "query": "how to sort a list",
            "tag": "python",
            "timestamp": (now - timedelta(days=3)).isoformat(),
            "result_count": 3
        },
        # Search 10 days ago (older)
        {
            "id": "5",
            "user_id": "user_4",
            "query": "git reset",
            "tag": "git",
            "timestamp": (now - timedelta(days=10)).isoformat(),
            "result_count": 2
        }
    ]
    
    mock_usage_history_storage.write_text(json.dumps(mock_history, indent=2), encoding="utf-8")
    mock_usage_history.documents.extend(mock_history)

    # 3. Test populated state
    resp2 = await client.get(f"{BASE}/analytics/trending")
    assert resp2.status_code == 200, resp2.text
    data2 = resp2.json()

    # active users today: user_1, user_2 (2 unique users)
    assert data2["active_users_today"] == 2
    
    # searches today: entries 1, 2, 3 (3 searches)
    assert data2["total_searches_today"] == 3
    
    # searches this week: entries 1, 2, 3, 4 (4 searches)
    assert data2["total_searches_week"] == 4

    # trending_queries:
    # "konek database" has 2 searches (tags: java, php)
    # "how to sort a list" has 2 searches (tags: python)
    # "git reset" has 1 search (tags: git)
    queries = data2["trending_queries"]
    assert len(queries) == 3
    
    q_map = {q["query"].lower(): q for q in queries}
    assert "konek database" in q_map
    assert q_map["konek database"]["count"] == 2
    assert q_map["konek database"]["tag"] in ("java", "php")

    assert "how to sort a list" in q_map
    assert q_map["how to sort a list"]["count"] == 2
    assert q_map["how to sort a list"]["tag"] == "python"

    # tag_distribution:
    # python: 2 searches
    # java: 1 search
    # php: 1 search
    # git: 1 search
    # total tag searches = 5
    tags = data2["tag_distribution"]
    assert len(tags) == 4
    
    tag_map = {t["tag"]: t for t in tags}
    assert "python" in tag_map
    assert tag_map["python"]["count"] == 2
    assert tag_map["python"]["percentage"] == 40.0
    
    assert "java" in tag_map
    assert tag_map["java"]["count"] == 1
    assert tag_map["java"]["percentage"] == 20.0






