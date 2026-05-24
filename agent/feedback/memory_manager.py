"""
Feedback Memory Manager
========================
Implements BOTH types of LangGraph memory as specified in the docs:

SHORT-TERM MEMORY  → InMemorySaver checkpointer (thread-scoped, per simulation session)
  - Tracks ongoing simulation runs within a session
  - Persists conversation history so a session can be resumed
  - Managed via thread_id → each simulation session is a unique thread

LONG-TERM MEMORY → InMemoryStore (cross-thread, user-scoped namespaces)
  - Semantic memory:   Twin persona profile + past simulation results (facts)
  - Episodic memory:   Past coaching interactions (experiences → few-shot examples)
  - Procedural memory: Coaching instructions that self-update after each session

Uses namespaced storage per the LangGraph docs:
  namespace = (user_id, "memories")        ← twin persona + facts
  namespace = (user_id, "episodes")        ← past simulation episodes
  namespace = (user_id, "instructions")    ← procedural coaching rules

References:
  https://docs.langchain.com/oss/python/langgraph/add-memory
  https://docs.langchain.com/oss/python/concepts/memory
"""

import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# ── LangGraph memory primitives ───────────────────────────────────
try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    from langgraph.graph import StateGraph, MessagesState, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# ── Optional: postgres-backed persistence for production ──────────
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.store.postgres import PostgresStore
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# ── Redis-backed store (free, as specified in architecture.md) ────
try:
    from langgraph.checkpoint.redis import RedisSaver
    REDIS_CHECKPOINTER_AVAILABLE = True
except ImportError:
    REDIS_CHECKPOINTER_AVAILABLE = False

# ── Gemini embeddings for semantic search in store ────────────────
try:
    from langchain.embeddings import init_embeddings
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


# ── Runtime context schema (per Deep Agents docs) ─────────────────
@dataclass
class FeedbackContext:
    """
    Runtime context injected at invoke time — follows Deep Agents pattern.
    Tools and nodes receive this via Runtime[FeedbackContext].
    """
    user_id: str
    session_id: str
    twin_persona_summary: str = ""
    total_simulations: int = 0


class FeedbackMemoryManager:
    """
    Central memory manager for the Feedback Engine.

    Provides:
      - get_checkpointer()  → short-term (thread-scoped) InMemorySaver
      - get_store()         → long-term (cross-thread) InMemoryStore with semantic search
      - save_twin_profile() → write to semantic memory namespace
      - save_episode()      → write coaching episode to episodic memory
      - load_past_episodes()→ retrieve few-shot examples for the coach
      - update_instructions()→ procedural memory: self-updating coaching rules
      - load_instructions() → retrieve current coaching rules for a user
      - search_similar_failures() → semantic search over past failures
    """

    # Fixed namespace keys (matches LangGraph store API)
    NS_MEMORIES    = "memories"       # semantic: facts about the twin
    NS_EPISODES    = "episodes"       # episodic: past simulation experiences
    NS_INSTRUCTIONS = "instructions"  # procedural: coaching rules

    def __init__(self):
        self._checkpointer: Optional[Any] = None
        self._store: Optional[Any] = None
        self._setup()

    def _setup(self):
        """
        Initialize the checkpointer and store.

        Priority order for checkpointer:
          1. Redis (free, fast — preferred per architecture.md)
          2. Postgres (production-grade)
          3. InMemorySaver (development fallback)

        Priority order for store:
          1. Postgres store (production)
          2. InMemoryStore with semantic search (development)
        """
        # ── Checkpointer (short-term / thread-scoped) ─────────────
        redis_url = os.getenv("REDIS_URL", "")
        pg_uri    = os.getenv("POSTGRES_URI", "")

        if REDIS_CHECKPOINTER_AVAILABLE and redis_url:
            try:
                self._checkpointer = RedisSaver.from_conn_string(redis_url)
                print("[FeedbackMemory] Short-term: Redis checkpointer ✓")
            except Exception as e:
                print(f"[FeedbackMemory] Redis checkpointer failed ({e}), falling back")

        if self._checkpointer is None and POSTGRES_AVAILABLE and pg_uri:
            try:
                self._checkpointer = PostgresSaver.from_conn_string(pg_uri)
                print("[FeedbackMemory] Short-term: Postgres checkpointer ✓")
            except Exception as e:
                print(f"[FeedbackMemory] Postgres checkpointer failed ({e}), falling back")

        if self._checkpointer is None and LANGGRAPH_AVAILABLE:
            self._checkpointer = InMemorySaver()
            print("[FeedbackMemory] Short-term: InMemorySaver (dev mode) ✓")

        # ── Store (long-term / cross-thread) ──────────────────────
        if POSTGRES_AVAILABLE and pg_uri:
            try:
                self._store = PostgresStore.from_conn_string(pg_uri)
                print("[FeedbackMemory] Long-term: PostgresStore ✓")
            except Exception as e:
                print(f"[FeedbackMemory] PostgresStore failed ({e}), falling back")

        if self._store is None and LANGGRAPH_AVAILABLE:
            # Use InMemoryStore with semantic search if embeddings are available
            if EMBEDDINGS_AVAILABLE:
                try:
                    embeddings = init_embeddings("google-genai:text-embedding-004")
                    self._store = InMemoryStore(
                        index={"embed": embeddings, "dims": 768}
                    )
                    print("[FeedbackMemory] Long-term: InMemoryStore + semantic search ✓")
                except Exception as e:
                    print(f"[FeedbackMemory] Semantic store failed ({e}), using plain store")

            if self._store is None:
                self._store = InMemoryStore()
                print("[FeedbackMemory] Long-term: InMemoryStore (plain) ✓")

    # ── Public accessors ──────────────────────────────────────────

    def get_checkpointer(self):
        """Returns the short-term memory checkpointer for graph compilation."""
        return self._checkpointer

    def get_store(self):
        """Returns the long-term memory store for graph compilation."""
        return self._store

    def new_thread_id(self) -> str:
        """Generate a unique thread_id for a new simulation session."""
        return f"sim-{uuid.uuid4().hex[:12]}"

    # ── Semantic Memory (facts about the twin) ────────────────────

    def save_twin_profile(self, user_id: str, persona: dict, video_analysis: dict):
        """
        Write the Digital Twin profile to long-term semantic memory.
        Namespace: (user_id, "memories"), key: "twin_profile"
        """
        if not self._store:
            return
        namespace = (user_id, self.NS_MEMORIES)
        profile_doc = {
            "type": "twin_profile",
            "persona_summary": persona.get("persona_summary", ""),
            "personality_dimensions": persona.get("personality_dimensions", {}),
            "communication_fingerprint": persona.get("communication_fingerprint", {}),
            "scenario_specific": persona.get("scenario_specific", {}),
            "embodied_signals": persona.get("embodied_signals", {}),
            "video_confidence_score": video_analysis.get(
                "body_language_analysis", {}
            ).get("average_metrics", {}).get("avg_confidence_score", 0),
            "dominant_emotion": self._dominant_emotion(
                video_analysis.get("facial_analysis", {}).get("emotion_distribution", {})
            ),
            "speaking_pace": video_analysis.get("voice_speech_analysis", {}).get(
                "audio_features", {}
            ).get("speaking_rate", {}).get("pace_label", ""),
        }
        self._store.put(namespace, "twin_profile", profile_doc)
        print(f"[FeedbackMemory] Saved twin profile for user={user_id}")

    def load_twin_profile(self, user_id: str) -> Optional[dict]:
        """Retrieve the stored twin profile from semantic memory."""
        if not self._store:
            return None
        namespace = (user_id, self.NS_MEMORIES)
        item = self._store.get(namespace, "twin_profile")
        return item.value if item else None

    def save_simulation_facts(self, user_id: str, sim_id: str, facts: dict):
        """
        Store key facts from a completed simulation run.
        Used as semantic memory for future coaching sessions.
        """
        if not self._store:
            return
        namespace = (user_id, self.NS_MEMORIES)
        self._store.put(namespace, f"sim_facts_{sim_id}", {
            "type": "simulation_facts",
            "sim_id": sim_id,
            "overall_success_rate": facts.get("overall_success_rate", 0),
            "category_results": facts.get("category_results", {}),
            "top_improvement_tags": facts.get("top_improvement_tags", {}),
            "critical_insight": facts.get("critical_insight", ""),
        })

    def search_similar_failures(self, user_id: str, query: str, limit: int = 5) -> List[dict]:
        """
        Semantic search over stored memories — finds similar past failure patterns.
        Leverages InMemoryStore.search() with vector similarity.
        """
        if not self._store:
            return []
        namespace = (user_id, self.NS_MEMORIES)
        try:
            results = self._store.search(namespace, query=query, limit=limit)
            return [r.value for r in results]
        except Exception as e:
            print(f"[FeedbackMemory] Semantic search failed: {e}")
            # Fallback: list all and return first N
            try:
                all_items = self._store.list(namespace)
                return [item.value for item in all_items[:limit]]
            except Exception:
                return []

    # ── Episodic Memory (past experiences / few-shot examples) ────

    def save_episode(
        self,
        user_id: str,
        session_id: str,
        cluster_analysis: dict,
        feedback: dict,
    ):
        """
        Save a completed coaching session as an episodic memory.
        These become few-shot examples for future coaching LLM calls.
        Namespace: (user_id, "episodes"), key: session_id
        """
        if not self._store:
            return
        namespace = (user_id, self.NS_EPISODES)
        episode = {
            "type": "coaching_episode",
            "session_id": session_id,
            "overall_success_rate": cluster_analysis.get("overall_success_rate", 0),
            "total_simulations": cluster_analysis.get("total_simulations", 0),
            "top_failure_clusters": cluster_analysis.get("failure_clusters", [])[:3],
            "critical_insight": cluster_analysis.get("critical_insight", ""),
            "top_improvements": cluster_analysis.get("top_improvements", [])[:3],
            "executive_summary": feedback.get("executive_summary", ""),
            "the_one_thing": feedback.get("the_one_thing", ""),
        }
        self._store.put(namespace, session_id, episode)
        print(f"[FeedbackMemory] Saved episode for user={user_id}, session={session_id}")

    def load_past_episodes(self, user_id: str, limit: int = 3) -> List[dict]:
        """
        Load recent coaching episodes as few-shot examples.
        Injected into the coaching LLM prompt as episodic context.
        """
        if not self._store:
            return []
        namespace = (user_id, self.NS_EPISODES)
        try:
            items = self._store.search(namespace, query="coaching session simulation results", limit=limit)
            return [item.value for item in items]
        except Exception:
            try:
                items = self._store.list(namespace)
                return [item.value for item in items[-limit:]]
            except Exception:
                return []

    # ── Procedural Memory (self-updating coaching instructions) ───

    def load_instructions(self, user_id: str) -> str:
        """
        Load the current coaching instructions from procedural memory.
        If none exist yet, return the default coaching rules.
        Namespace: (user_id, "instructions"), key: "coaching_rules"
        """
        if not self._store:
            return self._default_instructions()
        namespace = (user_id, self.NS_INSTRUCTIONS)
        item = self._store.get(namespace, "coaching_rules")
        if item:
            return item.value.get("instructions", self._default_instructions())
        return self._default_instructions()

    def update_instructions(
        self,
        user_id: str,
        current_instructions: str,
        session_feedback: dict,
        llm_client: Any,
        model_name: str,
    ) -> str:
        """
        Procedural memory: use Gemini to refine coaching instructions based
        on the session outcome — implements the 'Reflection / meta-prompting'
        pattern from the LangGraph memory docs.

        The agent sees its current instructions + the session outcome and
        generates improved instructions for the NEXT session.
        """
        if not self._store or not llm_client:
            return current_instructions

        prompt = (
            f"Current coaching instructions:\n{current_instructions}\n\n"
            f"Latest session outcome:\n"
            f"  Success rate: {session_feedback.get('headline_stats', {}).get('overall_success_rate', 0)}%\n"
            f"  Key insight: {session_feedback.get('the_one_thing', '')}\n"
            f"  Executive summary: {session_feedback.get('executive_summary', '')[:400]}\n\n"
            "Based on what worked and what failed this session, write improved coaching "
            "instructions for the next session. Be specific. Keep instructions under 500 words. "
            "Return only the updated instructions as plain text."
        )

        try:
            response = llm_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"temperature": 0.4},
            )
            new_instructions = response.text.strip()
            # Write back to procedural memory store
            namespace = (user_id, self.NS_INSTRUCTIONS)
            self._store.put(namespace, "coaching_rules", {"instructions": new_instructions})
            print(f"[FeedbackMemory] Updated coaching instructions for user={user_id}")
            return new_instructions
        except Exception as e:
            print(f"[FeedbackMemory] Instruction update failed: {e}")
            return current_instructions

    def delete_thread(self, thread_id: str):
        """Delete all checkpoints for a thread (clean up after session)."""
        if self._checkpointer:
            try:
                self._checkpointer.delete_thread(thread_id)
            except Exception as e:
                print(f"[FeedbackMemory] Thread delete failed: {e}")

    def get_thread_state(self, graph, thread_id: str) -> Optional[dict]:
        """View the current state snapshot for a thread (LangGraph docs pattern)."""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            snapshot = graph.get_state(config)
            return {
                "values": snapshot.values,
                "next": snapshot.next,
                "created_at": str(snapshot.created_at),
            }
        except Exception:
            return None

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _dominant_emotion(distribution: dict) -> str:
        if not distribution:
            return "neutral"
        return max(distribution, key=distribution.get)

    @staticmethod
    def _default_instructions() -> str:
        return """\
You are a world-class communication coach analyzing Digital Twin simulation results.

Core coaching principles:
1. Always cite specific numbers and percentages — never give vague feedback
2. Group failures by root cause, not surface symptom
3. Prioritize the top 1-2 improvements that will have the highest success rate impact
4. For each weakness, provide a concrete, repeatable daily drill
5. Compare personality type success rates to reveal hidden patterns
6. Deliver feedback with empathy but zero sugar-coating — be direct
7. Always end with a motivational but realistic closing statement
"""

    @property
    def is_available(self) -> bool:
        return LANGGRAPH_AVAILABLE and self._checkpointer is not None

    def info(self) -> dict:
        return {
            "langgraph_available": LANGGRAPH_AVAILABLE,
            "checkpointer_type": type(self._checkpointer).__name__ if self._checkpointer else "none",
            "store_type": type(self._store).__name__ if self._store else "none",
            "postgres_available": POSTGRES_AVAILABLE,
            "redis_checkpointer_available": REDIS_CHECKPOINTER_AVAILABLE,
            "embeddings_available": EMBEDDINGS_AVAILABLE,
        }


# Module-level singleton
_memory_manager: Optional[FeedbackMemoryManager] = None


def get_memory_manager() -> FeedbackMemoryManager:
    """Get the shared FeedbackMemoryManager instance (singleton)."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = FeedbackMemoryManager()
    return _memory_manager
