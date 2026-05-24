"""
User Service — JWT authentication + user profile CRUD.
Uses PostgreSQL in production, in-memory dict as dev fallback.
"""
import os
import uuid
import json
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, Dict

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

try:
    import psycopg2
    import psycopg2.extras
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False

# ── In-memory fallback store ──────────────────────────────────────
_users: Dict[str, dict] = {}
_by_email: Dict[str, str] = {}   # email → user_id

JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGO   = "HS256"
JWT_EXP_H  = 24  # hours


class UserService:
    """CRUD + authentication for user accounts."""

    def __init__(self):
        self._pg_conn = None
        if PG_AVAILABLE:
            uri = os.getenv("POSTGRES_URI", "")
            if uri:
                try:
                    self._pg_conn = psycopg2.connect(uri)
                    self._ensure_table()
                except Exception as e:
                    print(f"[UserService] Postgres unavailable ({e}), using in-memory store")

    # ── Table init ────────────────────────────────────────────────

    def _ensure_table(self):
        with self._pg_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id          TEXT PRIMARY KEY,
                    email       TEXT UNIQUE NOT NULL,
                    name        TEXT NOT NULL,
                    pw_hash     TEXT NOT NULL,
                    created_at  TIMESTAMP DEFAULT NOW(),
                    twin_id     TEXT,
                    meta        JSONB DEFAULT '{}'
                )
            """)
        self._pg_conn.commit()

    # ── Auth helpers ──────────────────────────────────────────────

    @staticmethod
    def _hash_password(password: str) -> str:
        salt = os.urandom(16).hex()
        h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
        return f"{salt}:{h.hex()}"

    @staticmethod
    def _verify_password(password: str, stored: str) -> bool:
        try:
            salt, h = stored.split(":", 1)
            check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
            return hmac.compare_digest(h, check.hex())
        except Exception:
            return False

    def _issue_token(self, user_id: str) -> str:
        if not JWT_AVAILABLE:
            return user_id  # fallback: just return user_id as "token"
        payload = {
            "sub": user_id,
            "exp": datetime.utcnow() + timedelta(hours=JWT_EXP_H),
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

    def verify_token(self, token: str) -> Optional[str]:
        """Return user_id if token valid, None otherwise."""
        if not JWT_AVAILABLE:
            return token  # passthrough in fallback mode
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
            return payload.get("sub")
        except Exception:
            return None

    # ── CRUD ─────────────────────────────────────────────────────

    def register(self, email: str, name: str, password: str) -> dict:
        email = email.lower().strip()
        if self._get_by_email(email):
            return {"error": "Email already registered"}
        user_id  = str(uuid.uuid4())
        pw_hash  = self._hash_password(password)
        user_doc = {
            "id": user_id,
            "email": email,
            "name": name,
            "pw_hash": pw_hash,
            "created_at": datetime.utcnow().isoformat(),
            "twin_id": None,
            "meta": {},
        }
        self._save_user(user_doc)
        token = self._issue_token(user_id)
        return {"user_id": user_id, "name": name, "email": email, "token": token}

    def login(self, email: str, password: str) -> dict:
        email = email.lower().strip()
        user  = self._get_by_email(email)
        if not user:
            return {"error": "Invalid credentials"}
        if not self._verify_password(password, user["pw_hash"]):
            return {"error": "Invalid credentials"}
        token = self._issue_token(user["id"])
        return {"user_id": user["id"], "name": user["name"], "email": email, "token": token}

    def get_user(self, user_id: str) -> Optional[dict]:
        user = self._load_user(user_id)
        if not user:
            return None
        return {k: v for k, v in user.items() if k != "pw_hash"}

    def set_twin_id(self, user_id: str, twin_id: str) -> bool:
        user = self._load_user(user_id)
        if not user:
            return False
        user["twin_id"] = twin_id
        self._save_user(user)
        return True

    # ── Storage backend ───────────────────────────────────────────

    def _save_user(self, user: dict):
        if self._pg_conn:
            with self._pg_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO users (id, email, name, pw_hash, twin_id, meta)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                    SET twin_id = EXCLUDED.twin_id, meta = EXCLUDED.meta
                """, (
                    user["id"], user["email"], user["name"],
                    user["pw_hash"], user.get("twin_id"),
                    json.dumps(user.get("meta", {}))
                ))
            self._pg_conn.commit()
        else:
            _users[user["id"]] = user
            _by_email[user["email"]] = user["id"]

    def _load_user(self, user_id: str) -> Optional[dict]:
        if self._pg_conn:
            with self._pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                row = cur.fetchone()
                return dict(row) if row else None
        return _users.get(user_id)

    def _get_by_email(self, email: str) -> Optional[dict]:
        if self._pg_conn:
            with self._pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM users WHERE email = %s", (email,))
                row = cur.fetchone()
                return dict(row) if row else None
        uid = _by_email.get(email)
        return _users.get(uid) if uid else None
