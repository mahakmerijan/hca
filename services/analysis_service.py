"""
Analysis Service — runs failure cluster analysis and generates coaching feedback.
Integrates: FailureClusterAnalyzer (Deep Agents harness) + FeedbackGenerator + memory.
"""
import os
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from typing import Optional, Dict, List

try:
    import psycopg2
    import psycopg2.extras
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False

from agent.feedback.cluster_analyzer import FailureClusterAnalyzer
from agent.feedback.feedback_generator import FeedbackGenerator
from agent.feedback.memory_manager import get_memory_manager
from agent.memory.redis_cache import RedisCache

# In-memory fallback
_analyses: Dict[str, dict] = {}


class AnalysisService:
    """
    Post-simulation analysis pipeline:
      1. FailureClusterAnalyzer → Deep Agents harness, LangGraph memory (short + long term)
      2. FeedbackGenerator      → structured coaching debrief (30-day plan, etc.)
      3. Saves episode to long-term episodic memory
      4. Updates procedural memory (reflection / meta-prompting)
      5. Caches result for fast retrieval
    """

    def __init__(self):
        self._pg_conn   = None
        self._cache     = RedisCache()
        self._analyzer  = FailureClusterAnalyzer()
        self._generator = FeedbackGenerator()
        self._memory    = get_memory_manager()

        if PG_AVAILABLE:
            uri = os.getenv("POSTGRES_URI", "")
            if uri:
                try:
                    self._pg_conn = psycopg2.connect(uri)
                    self._ensure_table()
                except Exception as e:
                    print(f"[AnalysisService] Postgres unavailable ({e}), using in-memory")

    def _ensure_table(self):
        with self._pg_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    analysis_id     TEXT PRIMARY KEY,
                    sim_id          TEXT NOT NULL,
                    user_id         TEXT NOT NULL,
                    cluster_report  JSONB,
                    feedback        JSONB,
                    created_at      TIMESTAMP DEFAULT NOW()
                )
            """)
        self._pg_conn.commit()

    # ── Public API ────────────────────────────────────────────────

    def run_analysis(
        self,
        sim_id: str,
        simulation_results: List[dict],
        user_id: str,
        twin_persona: dict,
    ) -> dict:
        """
        Full analysis pipeline for a completed simulation run.
        Returns analysis document with cluster report + coaching feedback.
        """
        # Check cache first
        cache_key = f"analysis:{sim_id}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        analysis_id = str(uuid.uuid4())
        session_id  = self._memory.new_thread_id()

        # Step 1: Cluster analysis (blocking — feedback depends on it)
        cluster_report = self._analyzer.analyze(
            simulation_results=simulation_results,
            user_id=user_id,
            session_id=session_id,
        )

        # Step 2: Feedback generation — runs concurrently with _save_analysis
        # Both can start immediately since feedback only needs cluster_report
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_feedback = ex.submit(
                self._generator.generate,
                cluster_report,
                twin_persona,
            )
            # Pre-build partial doc so saves can start right away
            analysis_id_val = analysis_id
            # Wait for feedback (the only blocking dep for the response)
            feedback = f_feedback.result() or {}

        analysis_doc = {
            "analysis_id":    analysis_id,
            "sim_id":         sim_id,
            "user_id":        user_id,
            "cluster_report": cluster_report,
            "feedback":       feedback,
            "created_at":     datetime.utcnow().isoformat(),
        }

        # All post-response work (persist + memory) runs in background threads
        # so the HTTP response is returned immediately after feedback is ready
        def _background_saves():
            try:
                self._save_analysis(analysis_doc)
                self._cache.set(cache_key, analysis_doc, ttl=7200)
                self._memory.save_episode(
                    user_id=user_id,
                    session_id=session_id,
                    cluster_analysis=cluster_report,
                    feedback=feedback,
                )
                self._update_procedural_memory(user_id, cluster_report, feedback)
            except Exception as e:
                print(f"[AnalysisService] Background save error: {e}")

        threading.Thread(target=_background_saves, daemon=True).start()

        return analysis_doc

    def get_analysis(self, analysis_id: str) -> Optional[dict]:
        cached = self._cache.get(f"analysis_id:{analysis_id}")
        if cached:
            return cached
        return self._load_analysis(analysis_id)

    def get_insights(self, user_id: str) -> dict:
        """
        Return aggregated insights across all analyses for this user.
        Pulls from long-term episodic memory + vector store.
        """
        episodes = self._memory.load_past_episodes(user_id, limit=10)
        instructions = self._memory.load_instructions(user_id)
        common_failures = self._memory.search_similar_failures(
            user_id, "most common failure patterns", limit=5
        )
        return {
            "user_id":          user_id,
            "total_sessions":   len(episodes),
            "past_episodes":    episodes,
            "coaching_rules":   instructions,
            "common_failures":  common_failures,
        }

    # ── Procedural memory update ──────────────────────────────────

    def _update_procedural_memory(self, user_id: str, cluster_report: dict, feedback: dict):
        """
        Reflection loop: distill top failure pattern into concise coaching rules.
        These rules are stored as procedural memory and injected into future analyses.
        """
        existing = self._memory.load_instructions(user_id) or ""
        top_clusters = cluster_report.get("failure_clusters", [])[:3]
        top_improvements = cluster_report.get("top_improvements", [])[:3]

        new_rules = []
        for c in top_clusters:
            rule = f"- Watch for '{c.get('cluster_name')}' ({c.get('frequency_pct', 0):.0f}%): {c.get('fix', '')}"
            new_rules.append(rule)
        for imp in top_improvements:
            rule = f"- Priority drill: {imp.get('improvement', '')} → {imp.get('practice_drill', '')}"
            new_rules.append(rule)

        if new_rules:
            updated = (existing + "\n" if existing else "") + "\n".join(new_rules)
            self._memory.update_instructions(user_id, updated[:3000])

    # ── Storage ───────────────────────────────────────────────────

    def _save_analysis(self, doc: dict):
        import json
        if self._pg_conn:
            with self._pg_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO analyses (analysis_id, sim_id, user_id, cluster_report, feedback)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (analysis_id) DO NOTHING
                """, (
                    doc["analysis_id"], doc["sim_id"], doc["user_id"],
                    json.dumps(doc["cluster_report"]),
                    json.dumps(doc["feedback"]),
                ))
            self._pg_conn.commit()
        else:
            _analyses[doc["analysis_id"]] = doc

    def _load_analysis(self, analysis_id: str) -> Optional[dict]:
        if self._pg_conn:
            with self._pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM analyses WHERE analysis_id = %s", (analysis_id,))
                row = cur.fetchone()
                return dict(row) if row else None
        return _analyses.get(analysis_id)
