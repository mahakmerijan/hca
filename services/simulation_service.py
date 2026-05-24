"""
Simulation Service — orchestrates the 10-run Digital Twin simulation.
Manages simulation state, progress tracking, and result storage.
"""
import os
import uuid
import json
from datetime import datetime
from typing import Optional, Dict, List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import psycopg2
    import psycopg2.extras
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False

from agent.simulation.scenario_generator import ScenarioGenerator
from agent.simulation.simulation_loop import SimulationLoop
from agent.memory.redis_cache import RedisCache

# In-memory fallback
_simulations: Dict[str, dict] = {}


class SimulationService:
    """
    Manages full 10-scenario simulation runs for a Digital Twin.

    Flow:
      1. Create simulation record (status: pending)
      2. Generate 10 scenarios (3 job interviews, 3 investor pitches, 4 dating)
    3. Run SimulationLoop (LangGraph StateGraph, batched, parallel workers)
      4. Stream progress via Redis pub/sub
      5. Store results + update status: completed
    """

    def __init__(self):
        self._pg_conn    = None
        self._cache      = RedisCache()
        self._scenario_gen = ScenarioGenerator()
        self._loop         = SimulationLoop()

        if PG_AVAILABLE:
            uri = os.getenv("POSTGRES_URI", "")
            if uri:
                try:
                    self._pg_conn = psycopg2.connect(uri)
                    self._ensure_tables()
                except Exception as e:
                    print(f"[SimulationService] Postgres unavailable ({e}), using in-memory")

    def _ensure_tables(self):
        with self._pg_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS simulations (
                    sim_id      TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    twin_id     TEXT NOT NULL,
                    status      TEXT DEFAULT 'pending',
                    total       INT  DEFAULT 0,
                    completed   INT  DEFAULT 0,
                    results     JSONB DEFAULT '[]',
                    created_at  TIMESTAMP DEFAULT NOW(),
                    updated_at  TIMESTAMP DEFAULT NOW()
                )
            """)
        self._pg_conn.commit()

    # ── Public API ────────────────────────────────────────────────

    def start_simulation(self, user_id: str, twin_id: str, twin_persona: dict) -> dict:
        """
        Create a simulation record and immediately kick off async execution.
        Returns sim_id + status=running.
        """
        sim_id = str(uuid.uuid4())
        scenarios = self._scenario_gen.generate_all()

        sim_doc = {
            "sim_id":     sim_id,
            "user_id":    user_id,
            "twin_id":    twin_id,
            "status":     "running",
            "total":      len(scenarios),
            "completed":  0,
            "results":    [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        self._save_simulation(sim_doc)
        self._cache.set(f"sim:active:{sim_id}", sim_doc, ttl=3600)

        # Launch in background thread
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(self._run_background, sim_id, scenarios, twin_persona)
        executor.shutdown(wait=False)

        return {"sim_id": sim_id, "status": "running", "total": len(scenarios)}

    def get_simulation(self, sim_id: str) -> Optional[dict]:
        # Try cache first (fastest)
        cached = self._cache.get(f"sim:active:{sim_id}")
        if cached:
            return cached
        return self._load_simulation(sim_id)

    def get_simulation_step(self, sim_id: str, step: int) -> Optional[dict]:
        """Return result for a single simulation step (0-indexed)."""
        sim = self.get_simulation(sim_id)
        if not sim:
            return None
        results = sim.get("results", [])
        if step < len(results):
            return results[step]
        return {"status": "not_yet_run", "step": step}

    def get_results(self, sim_id: str) -> List[dict]:
        sim = self.get_simulation(sim_id)
        return sim.get("results", []) if sim else []

    # ── Background execution ──────────────────────────────────────

    def _run_background(self, sim_id: str, scenarios: List[dict], twin_persona: dict):
        results = []

        def on_progress(completed_count: int, total: int, last_result: dict):
            results.append(last_result)
            sim = self._load_simulation(sim_id) or {}
            sim["completed"]  = completed_count
            sim["results"]    = list(results)
            sim["updated_at"] = datetime.utcnow().isoformat()
            # Store the latest conversation so the frontend can display it live
            sim["live_turn"] = {
                "scenario_num":  completed_count,
                "total":         total,
                "category":      last_result.get("category", ""),
                "counter_party": last_result.get("counter_party_name", "Agent"),
                "score":         last_result.get("overall_score", 0),
                "outcome":       last_result.get("verdict", ""),
                "conversation":  last_result.get("conversation", []),
            }
            if completed_count >= total:
                sim["status"] = "completed"
            self._save_simulation(sim)
            self._cache.set(f"sim:active:{sim_id}", sim, ttl=3600)
            self._cache.set(f"sim:progress:{sim_id}", {
                "sim_id": sim_id, "completed": completed_count, "total": total
            }, ttl=60)

        try:
            max_workers = int(os.getenv("SIM_MAX_WORKERS", "5"))
            self._loop.run_batch(
                scenarios,
                twin_persona=twin_persona,
                progress_callback=on_progress,
                max_workers=max_workers,
            )
            # Final status update
            sim = self._load_simulation(sim_id) or {}
            sim["status"]     = "completed"
            sim["updated_at"] = datetime.utcnow().isoformat()
            self._save_simulation(sim)
            self._cache.set(f"sim:active:{sim_id}", sim, ttl=3600)
        except Exception as e:
            sim = self._load_simulation(sim_id) or {}
            sim["status"] = "error"
            sim["error"]  = str(e)
            self._save_simulation(sim)
            self._cache.set(f"sim:active:{sim_id}", sim, ttl=3600)
            print(f"[SimulationService] Simulation {sim_id} error: {e}")

    # ── Storage ───────────────────────────────────────────────────

    def _save_simulation(self, doc: dict):
        if self._pg_conn:
            with self._pg_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO simulations (sim_id, user_id, twin_id, status, total, completed, results, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (sim_id) DO UPDATE
                    SET status = EXCLUDED.status, completed = EXCLUDED.completed,
                        results = EXCLUDED.results, updated_at = NOW()
                """, (
                    doc["sim_id"], doc["user_id"], doc["twin_id"],
                    doc["status"], doc["total"], doc["completed"],
                    json.dumps(doc.get("results", [])),
                ))
            self._pg_conn.commit()
        else:
            _simulations[doc["sim_id"]] = doc

    def _load_simulation(self, sim_id: str) -> Optional[dict]:
        if self._pg_conn:
            with self._pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM simulations WHERE sim_id = %s", (sim_id,))
                row = cur.fetchone()
                return dict(row) if row else None
        return _simulations.get(sim_id)
