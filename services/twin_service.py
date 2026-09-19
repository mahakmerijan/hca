"""
Twin Service — Digital Twin creation, storage, and retrieval.
Orchestrates: form schema → profile builder → persona generator → memory store.
"""
import os
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

try:
    import psycopg2
    import psycopg2.extras
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False

from agent.twin.profile_builder import TwinProfileBuilder
from agent.twin.persona_generator import PersonaGenerator
from agent.feedback.memory_manager import get_memory_manager

# In-memory fallback
_twins: Dict[str, dict] = {}
# File-based persistence (survives server restarts when no Postgres)
_TWINS_FILE = Path(__file__).parent.parent / "output" / "twins_store.json"

def _load_twins_from_file():
    """Load persisted twins from disk into the in-memory dict on startup."""
    if _TWINS_FILE.exists():
        try:
            data = json.loads(_TWINS_FILE.read_text())
            _twins.update(data)
            print(f"[TwinService] Loaded {len(data)} twin(s) from disk")
        except Exception as e:
            print(f"[TwinService] Could not load twins file: {e}")

_load_twins_from_file()


class TwinService:
    """
    Manages the full Digital Twin lifecycle:
      1. Validate + merge form answers with video analysis
      2. Build structured profile (TwinProfileBuilder)
      3. Generate LLM persona (PersonaGenerator → Gemini 2.5 Pro)
      4. Store in long-term memory (InMemoryStore / PostgresStore)
      5. Return twin_id for downstream simulation use
    """

    def __init__(self):
        self._pg_conn     = None
        self._profile_builder = TwinProfileBuilder()
        self._persona_gen     = PersonaGenerator()
        self._memory          = get_memory_manager()

        if PG_AVAILABLE:
            uri = os.getenv("POSTGRES_URI", "")
            if uri:
                try:
                    self._pg_conn = psycopg2.connect(uri)
                    self._ensure_table()
                except Exception as e:
                    print(f"[TwinService] Postgres unavailable ({e}), using in-memory")

    def _ensure_table(self):
        with self._pg_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS digital_twins (
                    twin_id     TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    user_name   TEXT,
                    profile     JSONB,
                    persona     JSONB,
                    created_at  TIMESTAMP DEFAULT NOW(),
                    updated_at  TIMESTAMP DEFAULT NOW()
                )
            """)
        self._pg_conn.commit()

    # ── Public API ────────────────────────────────────────────────

    def create_twin(
        self,
        user_id: str,
        user_name: str,
        form_data: dict,
        video_analysis: Optional[dict] = None,
    ) -> dict:
        """
        Full Digital Twin creation pipeline.
        Returns: { twin_id, user_id, persona, profile, created_at }
        """
        twin_id = str(uuid.uuid4())

        # 1. Merge form + video into structured profile
        profile = self._profile_builder.build(
            form_data=form_data,
            video_analysis=video_analysis or {},
            user_name=user_name,
        )

        # 2. Generate LLM persona (Gemini 2.5 Pro)
        persona = self._persona_gen.generate(profile)

        twin_doc = {
            "twin_id":    twin_id,
            "user_id":    user_id,
            "user_name":  user_name,
            "profile":    profile,
            "persona":    persona,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        # 3. Save to storage + long-term memory
        self._save_twin(twin_doc)
        self._memory.save_twin_profile(user_id, persona, video_analysis or {})

        return twin_doc

    def get_twin(self, twin_id: str) -> Optional[dict]:
        return self._load_twin(twin_id)

    def get_twin_for_user(self, user_id: str) -> Optional[dict]:
        """Return the latest twin belonging to this user."""
        if self._pg_conn:
            with self._pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM digital_twins WHERE user_id = %s ORDER BY created_at DESC LIMIT 1",
                    (user_id,),
                )
                row = cur.fetchone()
                return dict(row) if row else None
        # In-memory fallback: linear scan
        candidates = [t for t in _twins.values() if t["user_id"] == user_id]
        return max(candidates, key=lambda t: t["created_at"]) if candidates else None

    def update_twin(self, twin_id: str, form_data: dict, video_analysis: Optional[dict] = None) -> dict:
        existing = self._load_twin(twin_id)
        if not existing:
            return {"error": "Twin not found"}
        updated_profile = self._profile_builder.build(
            form_data=form_data,
            video_analysis=video_analysis or existing.get("profile", {}).get("video_analysis", {}),
            user_name=existing["user_name"],
        )
        updated_persona = self._persona_gen.generate(updated_profile)
        existing.update({
            "profile":    updated_profile,
            "persona":    updated_persona,
            "updated_at": datetime.utcnow().isoformat(),
        })
        self._save_twin(existing)
        self._memory.save_twin_profile(existing["user_id"], updated_persona, video_analysis or {})
        return existing

    # ── Storage ───────────────────────────────────────────────────

    def _save_twin(self, doc: dict):
        if self._pg_conn:
            with self._pg_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO digital_twins (twin_id, user_id, user_name, profile, persona, updated_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (twin_id) DO UPDATE
                    SET profile = EXCLUDED.profile, persona = EXCLUDED.persona, updated_at = NOW()
                """, (
                    doc["twin_id"], doc["user_id"], doc["user_name"],
                    json.dumps(doc["profile"]), json.dumps(doc["persona"]),
                ))
            self._pg_conn.commit()
        else:
            _twins[doc["twin_id"]] = doc
            # Persist to disk so twins survive server restarts
            try:
                _TWINS_FILE.parent.mkdir(parents=True, exist_ok=True)
                _TWINS_FILE.write_text(json.dumps(_twins, default=str))
            except Exception as e:
                print(f"[TwinService] Could not persist twins to disk: {e}")

    def _load_twin(self, twin_id: str) -> Optional[dict]:
        if self._pg_conn:
            with self._pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM digital_twins WHERE twin_id = %s", (twin_id,))
                row = cur.fetchone()
                if not row:
                    return None
                result = dict(row)
                # psycopg2 returns JSONB as dict already
                return result
        return _twins.get(twin_id)
