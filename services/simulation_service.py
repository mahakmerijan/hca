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
from agent.token_logger import log_run_summary, run_context, track_stage

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

    def start_simulation(
        self,
        user_id: str,
        twin_id: str,
        twin_persona: dict,
        mode: str = "predefined",
        custom_persona: Optional[dict] = None,
    ) -> dict:
        """
        Create a simulation record and immediately kick off async execution.

        Args:
            mode: "predefined" | "custom" | "both"
              - predefined: run the 10 built-in scenarios
              - custom:     run only the custom persona (1 scenario, 50 turns)
              - both:       run predefined 10 + custom persona

        Returns sim_id + status=running.
        """
        sim_id = str(uuid.uuid4())

        # Determine which scenarios to run
        if mode in ("predefined", "both"):
            scenarios = self._scenario_gen.generate_all()
        else:
            scenarios = []

        total = len(scenarios) + (1 if mode in ("custom", "both") and custom_persona else 0)

        sim_doc = {
            "sim_id":          sim_id,
            "user_id":         user_id,
            "twin_id":         twin_id,
            "mode":            mode,
            "status":          "running",
            "total":           total,
            "completed":       0,
            "results":         [],
            "custom_result":   None,
            "created_at":      datetime.utcnow().isoformat(),
            "started_at":      datetime.utcnow().isoformat() + "Z",
            "updated_at":      datetime.utcnow().isoformat(),
        }
        self._save_simulation(sim_doc)
        self._cache.set(f"sim:active:{sim_id}", sim_doc, ttl=3600)

        # Launch in background thread
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(self._run_background, sim_id, scenarios, twin_persona, mode, custom_persona)
        executor.shutdown(wait=False)

        return {"sim_id": sim_id, "status": "running", "total": total, "mode": mode}

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
        if not sim:
            return []
        results = list(sim.get("results") or [])
        # Include custom_result as a regular result entry so analysis sees it
        cr = sim.get("custom_result")
        if cr and isinstance(cr, dict):
            entry = dict(cr)
            entry.setdefault("scenario_num", len(results) + 1)
            entry.setdefault("category", cr.get("category", "custom"))
            results.append(entry)
        return results

    # ── Background execution ──────────────────────────────────────

    def _run_background(
        self,
        sim_id: str,
        scenarios: List[dict],
        twin_persona: dict,
        mode: str = "predefined",
        custom_persona: Optional[dict] = None,
    ):
        results = []
        sim = self._load_simulation(sim_id) or {}
        telemetry_run = run_context(sim_id, "simulation", sim.get("user_id", ""))
        telemetry_run.__enter__()

        def on_progress(completed_count: int, total: int, last_result: dict):
            results.append(last_result)
            print(f"[SimulationService] {sim_id[:8]} progress {completed_count}/{total} — scenario {last_result.get('category','?')} score={last_result.get('overall_score','?')}", flush=True)
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
            self._save_simulation(sim)
            self._cache.set(f"sim:active:{sim_id}", sim, ttl=3600)
            self._cache.set(f"sim:progress:{sim_id}", {
                "sim_id": sim_id, "completed": completed_count, "total": total
            }, ttl=60)

        try:
            print(f"[SimulationService] Starting background run for {sim_id}, mode={mode}, scenarios={len(scenarios)}", flush=True)
            # Run predefined scenarios
            if mode in ("predefined", "both") and scenarios:
                max_workers = int(os.getenv("SIM_MAX_WORKERS", "5"))
                total = len(scenarios) + (1 if mode == "both" and custom_persona else 0)
                with track_stage("simulation_batch", "Gemini multi-agent simulation", "llm", {"scenarios": len(scenarios)}):
                    self._loop.run_batch(
                        scenarios,
                        twin_persona=twin_persona,
                        progress_callback=lambda c, t, r: on_progress(c, total, r),
                        max_workers=max_workers,
                    )

            # Run custom persona simulation
            if mode in ("custom", "both") and custom_persona:
                predefined_done = len(results)
                total = predefined_done + 1
                sim = self._load_simulation(sim_id) or {}
                sim["status"]     = "running_custom"
                sim["updated_at"] = datetime.utcnow().isoformat()
                self._save_simulation(sim)
                self._cache.set(f"sim:active:{sim_id}", sim, ttl=3600)

                print(f"[SimulationService] Starting custom persona simulation (50 turns)", flush=True)
                with track_stage("custom_persona_simulation", "Gemini multi-agent simulation", "llm", {"max_turns": 50}):
                    custom_result = self._loop.run_custom_persona(
                        custom_persona=custom_persona,
                        twin_persona=twin_persona,
                        max_turns=50,
                    )
                sim = self._load_simulation(sim_id) or {}
                sim["custom_result"] = custom_result
                sim["completed"]     = predefined_done + 1
                sim["updated_at"]    = datetime.utcnow().isoformat()
                self._save_simulation(sim)
                self._cache.set(f"sim:active:{sim_id}", sim, ttl=3600)

            # Final status update
            sim = self._load_simulation(sim_id) or {}
            sim["status"]     = "completed"
            sim["updated_at"] = datetime.utcnow().isoformat()
            self._save_simulation(sim)
            self._cache.set(f"sim:active:{sim_id}", sim, ttl=3600)
            # Write run-level token summary
            log_run_summary(
                run_type="simulation",
                sim_id=sim_id,
                user_id=sim.get("user_id"),
                started_at=sim.get("started_at"),
            )

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            sim = self._load_simulation(sim_id) or {}
            sim["status"] = "error"
            sim["error"]  = str(e)
            sim["traceback"] = tb
            self._save_simulation(sim)
            self._cache.set(f"sim:active:{sim_id}", sim, ttl=3600)
            log_run_summary(run_type="simulation_error", sim_id=sim_id, user_id=sim.get("user_id"), started_at=sim.get("started_at"))
            print(f"[SimulationService] Simulation {sim_id} error: {e}\n{tb}", flush=True)
        finally:
            telemetry_run.__exit__(None, None, None)

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
