"""
Redis Cache
===========
Handles:
  - Session data (active job states)
  - Active simulation state storage
  - Gemini context cache references
  - Fast twin profile retrieval

Falls back to in-memory dict if Redis is not available.
"""

import json
import os
import time
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisCache:
    """
    Unified Redis cache layer with in-memory fallback.
    Used for session data, simulation state, and fast retrieval.
    """

    # TTL constants (seconds)
    TTL_SESSION = 3600         # 1 hour for sessions
    TTL_SIMULATION = 900       # 15 min for active simulations
    TTL_TWIN = 86400           # 24 hours for twin profiles
    TTL_GEMINI_CACHE = 900     # 15 min — matches Gemini context cache TTL

    def __init__(self):
        self._mem: dict = {}   # fallback in-memory store
        self._mem_ttl: dict = {}
        self.client: Optional[Any] = None
        self.available = False
        self._connect()

    def _connect(self):
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        if not REDIS_AVAILABLE:
            print("[RedisCache] redis-py not installed — using in-memory fallback")
            return
        try:
            self.client = redis.from_url(url, decode_responses=True, socket_timeout=2)
            self.client.ping()
            self.available = True
            print(f"[RedisCache] Connected — {url}")
        except Exception as e:
            print(f"[RedisCache] Could not connect ({e}) — using in-memory fallback")
            self.client = None

    # ── Core get/set/delete ───────────────────────────────────────

    def set(self, key: str, value: Any, ttl: int = TTL_SESSION) -> bool:
        serialized = json.dumps(value, default=str)
        if self.available:
            try:
                self.client.setex(key, ttl, serialized)
                return True
            except Exception as e:
                print(f"[RedisCache] Set error: {e}")
        # fallback
        self._mem[key] = serialized
        self._mem_ttl[key] = time.time() + ttl
        return True

    def get(self, key: str) -> Optional[Any]:
        if self.available:
            try:
                raw = self.client.get(key)
                return json.loads(raw) if raw else None
            except Exception as e:
                print(f"[RedisCache] Get error: {e}")
        # fallback
        if key in self._mem:
            if time.time() < self._mem_ttl.get(key, 0):
                return json.loads(self._mem[key])
            else:
                del self._mem[key]
        return None

    def delete(self, key: str) -> bool:
        if self.available:
            try:
                self.client.delete(key)
                return True
            except Exception:
                pass
        self._mem.pop(key, None)
        self._mem_ttl.pop(key, None)
        return True

    def exists(self, key: str) -> bool:
        if self.available:
            try:
                return bool(self.client.exists(key))
            except Exception:
                pass
        return key in self._mem and time.time() < self._mem_ttl.get(key, 0)

    # ── Domain-specific helpers ───────────────────────────────────

    def save_session(self, job_id: str, data: dict):
        self.set(f"session:{job_id}", data, self.TTL_SESSION)

    def get_session(self, job_id: str) -> Optional[dict]:
        return self.get(f"session:{job_id}")

    def save_twin(self, user_id: str, persona: dict):
        self.set(f"twin:{user_id}", persona, self.TTL_TWIN)

    def get_twin(self, user_id: str) -> Optional[dict]:
        return self.get(f"twin:{user_id}")

    def save_simulation_state(self, sim_id: str, state: dict):
        self.set(f"sim:{sim_id}", state, self.TTL_SIMULATION)

    def get_simulation_state(self, sim_id: str) -> Optional[dict]:
        return self.get(f"sim:{sim_id}")

    def update_simulation_progress(self, sim_id: str, completed: int, total: int, results: list):
        state = self.get_simulation_state(sim_id) or {}
        state.update({
            "completed": completed,
            "total": total,
            "progress_pct": round(completed / total * 100, 1) if total else 0,
            "partial_results": results[-10:],  # keep last 10 for live preview
        })
        self.save_simulation_state(sim_id, state)

    def save_gemini_cache_ref(self, user_id: str, cache_name: str):
        """Store Gemini explicit context cache name for the 10-sim run."""
        self.set(f"gemini_cache:{user_id}", {"cache_name": cache_name}, self.TTL_GEMINI_CACHE)

    def get_gemini_cache_ref(self, user_id: str) -> Optional[str]:
        data = self.get(f"gemini_cache:{user_id}")
        return data.get("cache_name") if data else None

    def save_analysis_result(self, sim_id: str, analysis: dict):
        """Store final cluster analysis result."""
        self.set(f"analysis:{sim_id}", analysis, self.TTL_TWIN)

    def get_analysis_result(self, sim_id: str) -> Optional[dict]:
        return self.get(f"analysis:{sim_id}")

    def get_info(self) -> dict:
        info = {"backend": "redis" if self.available else "in_memory"}
        if self.available:
            try:
                server_info = self.client.info("server")
                info["redis_version"] = server_info.get("redis_version", "?")
                info["connected_clients"] = self.client.info("clients").get("connected_clients", 0)
            except Exception:
                pass
        else:
            info["cached_keys"] = len(self._mem)
        return info
