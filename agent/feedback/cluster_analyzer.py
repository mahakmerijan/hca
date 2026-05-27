"""
Failure Cluster Analyzer — Deep Agents Harness + LangGraph Memory
==================================================================
Implements the architecture.md Section 4 "Analysis & Feedback" using:

1. Deep Agents SDK (create_deep_agent) as the "agent harness"
     - system_prompt  → static coaching role definition
     - runtime context → user_id, session_id, simulation metadata
     - subagent        → isolates the heavy LLM cluster analysis (context isolation)
     - summarization   → auto-compacts context when window fills (deepagents middleware)

2. SHORT-TERM memory (InMemorySaver checkpointer)
     - Each analysis session is a unique thread_id
     - Allows session resumption if interrupted mid-analysis

3. LONG-TERM memory (InMemoryStore)
     - Semantic:    twin profile + past sim facts (vector-searchable)
     - Episodic:    past coaching sessions as few-shot examples
     - Procedural:  self-updating coaching instructions (reflection/meta-prompting)

References:
  https://docs.langchain.com/oss/python/deepagents/overview
  https://docs.langchain.com/oss/python/deepagents/context-engineering
  https://docs.langchain.com/oss/python/langgraph/add-memory
  https://docs.langchain.com/oss/python/concepts/memory
"""

import json
import os
import re
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# ── Deep Agents SDK ───────────────────────────────────────────────
try:
    from deepagents import create_deep_agent, SubAgent, SubAgentMiddleware
    from deepagents.middleware.summarization import create_summarization_middleware
    DEEPAGENTS_AVAILABLE = True
except ImportError:
    DEEPAGENTS_AVAILABLE = False

# ── LangGraph primitives ──────────────────────────────────────────
try:
    from langgraph.graph import StateGraph, END
    from langchain.tools import tool, ToolRuntime
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# ── LangChain model wrapper (Vertex AI) ──────────────────────────
try:
    from langchain_google_vertexai import ChatVertexAI
    LANGCHAIN_GOOGLE_AVAILABLE = True
except ImportError:
    LANGCHAIN_GOOGLE_AVAILABLE = False

# ── Raw Gemini SDK fallback ───────────────────────────────────────
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from .memory_manager import FeedbackContext, FeedbackMemoryManager, get_memory_manager


# ═══════════════════════════════════════════════════════════════════
# SYSTEM PROMPT  (static input context — per Deep Agents docs)
# ═══════════════════════════════════════════════════════════════════

HARNESS_SYSTEM_PROMPT = """\
You are a world-class communication coach and behavioural analyst.
You analyze results from 10 social-interaction simulations run by a
user's Digital Twin and produce a precise, data-driven coaching debrief.

Core rules:
1. Group failures by ROOT CAUSE — never give random surface-level feedback
2. Find the exact conversation stage where each failure cluster occurs
3. Identify personality types the twin succeeds/fails with and explain WHY
4. Produce specific, repeatable improvement drills (not vague advice)
5. Cite exact numbers: "In 73% of your failed investor pitches..."
6. Write in second person ("you", "your") — direct and empathetic

When delegating heavy analysis work, use the task tool to spawn a subagent.
The subagent must return only a concise JSON summary, not raw data.
"""

CLUSTER_ANALYSIS_PROMPT = """\
You are an expert communication coach and behavioural scientist.
Analyze the simulation statistics AND the actual conversation turns provided.

For each failure, go deep: identify the EXACT exchange where it went wrong,
WHY it went wrong (psychologically and linguistically), and what the twin
should have said instead.

Return ONLY valid JSON (no markdown fences):
{
  "failure_clusters": [
    {
      "cluster_name": "Short name (e.g. Risk Deflection)",
      "frequency_pct": <0-100>,
      "description": "What this failure pattern looks like in conversation — quote actual twin lines where possible",
      "root_cause": "Deep psychological/behavioral root cause — WHY does this person do this?",
      "affected_categories": ["job_interview"],
      "typical_moment": "opening | reaction | curveball | closing",
      "exact_breakdown_exchange": "Describe the specific turn-by-turn moment where it collapsed: what the counter-party said, how the twin responded, and why that response failed",
      "what_twin_said": "Example of actual or typical twin response that caused the failure",
      "what_should_have_been_said": "The ideal alternative response the twin should have given",
      "fix": "Precise 2-3 sentence actionable fix"
    }
  ],
  "personality_matrix": [
    {
      "personality_type": "Analytical",
      "success_rate": <0-100>,
      "sample_size": <n>,
      "why": "Why the twin performs this way with this personality type — specific behavioral mismatch",
      "strategy": "How to approach this type better — concrete behavioral changes"
    }
  ],
  "critical_insight": "The single most important pattern across all runs — be specific and direct.",
  "success_highlights": ["Specific things the twin did well in successful runs — quote exchanges if possible"],
  "top_improvements": [
    {
      "rank": 1,
      "improvement": "Specific behavior to change",
      "why_it_matters": "Why this was costing them in conversations",
      "expected_impact": "+X% estimated success rate lift",
      "practice_drill": "Concrete daily drill with example script"
    }
  ]
}
"""


# ═══════════════════════════════════════════════════════════════════
# LANGGRAPH STATE (used by fallback graph)
# ═══════════════════════════════════════════════════════════════════

if LANGGRAPH_AVAILABLE:
    from typing import TypedDict

    class ClusterState(TypedDict):
        raw_results: List[dict]
        stats: dict
        improvement_tags_freq: dict
        personality_matrix: dict
        past_episodes: List[dict]         # episodic memory: few-shot examples
        coaching_instructions: str         # procedural memory: current rules
        similar_past_failures: List[dict]  # semantic memory: past failures
        llm_analysis: Optional[dict]
        final_report: Optional[dict]


# ═══════════════════════════════════════════════════════════════════
# TOOLS exposed to the Deep Agents harness
# ═══════════════════════════════════════════════════════════════════

if LANGGRAPH_AVAILABLE:
    @tool
    def retrieve_past_failures(query: str, runtime: "ToolRuntime[FeedbackContext]") -> str:
        """
        Search long-term memory for similar past failure patterns using semantic search.
        Use when you need context from previous simulation sessions for this user.
        Returns a JSON list of relevant past failure summaries.

        Args:
            query: Semantic search query describing the failure pattern to look for
        """
        memory = get_memory_manager()
        results = memory.search_similar_failures(runtime.context.user_id, query, limit=5)
        return json.dumps(results, default=str)

    @tool
    def get_twin_profile(runtime: "ToolRuntime[FeedbackContext]") -> str:
        """
        Retrieve the Digital Twin's long-term profile from semantic memory.
        Use to understand the user's baseline personality before analyzing failures.
        Returns JSON with the twin's profile facts.

        Args: none
        """
        memory = get_memory_manager()
        profile = memory.load_twin_profile(runtime.context.user_id)
        return json.dumps(profile or {}, default=str)

    @tool
    def get_past_coaching_episodes(runtime: "ToolRuntime[FeedbackContext]") -> str:
        """
        Retrieve past coaching session summaries from episodic memory.
        Use these as few-shot examples to improve analysis quality.
        Returns a JSON list of past episode summaries.

        Args: none
        """
        memory = get_memory_manager()
        episodes = memory.load_past_episodes(runtime.context.user_id, limit=3)
        return json.dumps(episodes, default=str)


# ═══════════════════════════════════════════════════════════════════
# MAIN CLASS
# ═══════════════════════════════════════════════════════════════════

class FailureClusterAnalyzer:
    """
    Deep Agents harness for post-simulation failure cluster analysis.

    Harness structure (when deepagents + langgraph available):
      create_deep_agent
        ├── checkpointer  → InMemorySaver  (short-term, thread-scoped)
        ├── store         → InMemoryStore  (long-term, cross-thread namespaces)
        ├── tools         → retrieve_past_failures, get_twin_profile, get_past_coaching_episodes
        ├── middleware    → SummarizationMiddleware (auto-compact at 85% context)
        └── subagent      → cluster_analyst (context isolation for heavy LLM work)

    Fallback: plain LangGraph StateGraph (still uses both memory types).
    Final fallback: direct Gemini SDK call.
    """

    def __init__(self):
        self.model_name = os.getenv("LLM_MODEL", "gemini-2.5-pro")
        self._project  = os.getenv("VERTEX_PROJECT", "ai-ml-integrations")
        self._location = os.getenv("VERTEX_LOCATION", "us-central1")
        self.memory     = get_memory_manager()

        # Raw Gemini client (for stats LLM calls)
        self._gemini = None
        if GENAI_AVAILABLE:
            self._gemini = genai.Client(
                vertexai=True,
                project=self._project,
                location=self._location,
            )

        # LangChain Vertex AI model (for Deep Agents harness)
        self._lc_model = None
        if LANGCHAIN_GOOGLE_AVAILABLE:
            try:
                self._lc_model = ChatVertexAI(
                    model=self.model_name,
                    project=self._project,
                    location=self._location,
                    temperature=0.4,
                )
            except Exception as e:
                print(f"[ClusterAnalyzer] LangChain Vertex AI model: {e}")

        self._deep_agent    = self._build_deep_agent_harness()
        self._fallback_graph = (
            self._build_fallback_graph()
            if not self._deep_agent and LANGGRAPH_AVAILABLE
            else None
        )

    # ── Harness construction ──────────────────────────────────────

    def _build_deep_agent_harness(self):
        """
        Build the Deep Agents harness per the docs:
        - static system_prompt (role)
        - runtime context schema (FeedbackContext)
        - memory tools (semantic + episodic retrieval)
        - summarization middleware (85% trigger, keeps 10%)
        - cluster_analyst subagent (context isolation)
        - checkpointer = short-term memory
        - store        = long-term memory
        """
        if not DEEPAGENTS_AVAILABLE or not self._lc_model:
            return None

        try:
            # Cluster analysis subagent — isolates heavy work.
            # Main agent stays clean; subagent returns only concise JSON.
            cluster_subagent = SubAgent(
                name="cluster_analyst",
                description=(
                    "Performs deep statistical failure cluster analysis. "
                    "Returns only a concise JSON summary — not raw data."
                ),
                system_prompt=(
                    CLUSTER_ANALYSIS_PROMPT
                    + "\nIMPORTANT: Return only the essential JSON (< 800 words). "
                    "Do NOT include raw simulation data in your response."
                ),
                model=self._lc_model,
            )

            # Summarization middleware — auto-compacts long analysis context
            summ_mw = create_summarization_middleware(model=self._lc_model)

            tools = []
            if LANGGRAPH_AVAILABLE:
                tools = [retrieve_past_failures, get_twin_profile, get_past_coaching_episodes]

            agent = create_deep_agent(
                model=self._lc_model,
                system_prompt=HARNESS_SYSTEM_PROMPT,
                tools=tools,
                middleware=[
                    summ_mw,
                    SubAgentMiddleware(subagents=[cluster_subagent]),
                ],
                context_schema=FeedbackContext,
                checkpointer=self.memory.get_checkpointer(),  # short-term memory
                store=self.memory.get_store(),                 # long-term memory
            )
            print("[ClusterAnalyzer] Deep Agents harness ready ✓")
            return agent
        except Exception as e:
            print(f"[ClusterAnalyzer] Deep Agents harness failed ({e}), using graph fallback")
            return None

    def _build_fallback_graph(self):
        """
        Plain LangGraph StateGraph fallback — still wires up both memory types.
        Short-term: checkpointer (thread_id → resumable session)
        Long-term:  store        (namespaced facts/episodes/instructions)
        """
        if not LANGGRAPH_AVAILABLE:
            return None

        graph = StateGraph(ClusterState)
        graph.add_node("load_memory",              self._node_load_memory)
        graph.add_node("compute_stats",            self._node_compute_stats)
        graph.add_node("analyze_tags",             self._node_analyze_tags)
        graph.add_node("build_personality_matrix", self._node_personality_matrix)
        graph.add_node("llm_cluster_analysis",     self._node_llm_analysis)
        graph.add_node("compile_report",           self._node_compile_report)

        graph.set_entry_point("load_memory")
        graph.add_edge("load_memory",              "compute_stats")
        graph.add_edge("compute_stats",            "analyze_tags")
        graph.add_edge("analyze_tags",             "build_personality_matrix")
        graph.add_edge("build_personality_matrix", "llm_cluster_analysis")
        graph.add_edge("llm_cluster_analysis",     "compile_report")
        graph.add_edge("compile_report",           END)

        return graph.compile(
            checkpointer=self.memory.get_checkpointer(),  # short-term
            store=self.memory.get_store(),                 # long-term
        )

    # ── Public API ────────────────────────────────────────────────

    def analyze(
        self,
        simulation_results: List[dict],
        user_id: str = "default",
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Run full cluster analysis pipeline on simulation results.

        Short-term memory: each call uses a unique thread_id → resumable.
        Long-term memory:  loads past episodes/profile; saves facts after run.
        """
        if session_id is None:
            session_id = self.memory.new_thread_id()

        # Run all independent work concurrently:
        #   - load_instructions, load_past_episodes (memory I/O)
        #   - _compute_stats, _compute_tags, _compute_personality_matrix (CPU)
        with ThreadPoolExecutor(max_workers=5) as ex:
            f_instructions = ex.submit(self.memory.load_instructions, user_id)
            f_episodes     = ex.submit(self.memory.load_past_episodes, user_id)
            f_stats        = ex.submit(self._compute_stats, simulation_results)
            f_tags         = ex.submit(self._compute_tags, simulation_results)
            f_pm           = ex.submit(self._compute_personality_matrix, simulation_results)

        instructions  = f_instructions.result()
        past_episodes = f_episodes.result()
        stats         = f_stats.result()
        tags          = f_tags.result()
        pm            = f_pm.result()

        if self._deep_agent:
            report = self._run_via_harness(
                simulation_results, stats, tags, pm,
                user_id, session_id, instructions, past_episodes,
            )
        elif self._fallback_graph:
            report = self._run_via_graph(
                simulation_results, stats, tags, pm,
                user_id, session_id, instructions, past_episodes,
            )
        else:
            report = self._run_direct_llm(
                simulation_results, stats, tags, pm,
                coaching_instructions=instructions,
                past_episodes=past_episodes,
            )

        return report

    # ── Deep Agents harness runner ────────────────────────────────

    def _run_via_harness(self, results, stats, tags, pm,
                         user_id, session_id, instructions, past_episodes) -> dict:
        failures = [r for r in results if r.get("verdict") == "failure"][:5]
        failure_samples = [
            {
                "category":            f.get("category"),
                "counter_party":       f.get("counter_party_name"),
                "personality":         f.get("counter_party_personality"),
                "score":               f.get("overall_score"),
                "tags":                f.get("improvement_tags", []),
                "key_failure_moment":  f.get("key_failure_moment"),
            }
            for f in failures
        ]

        # Build the message with all context layers injected:
        # 1. Procedural memory (coaching instructions)
        # 2. Episodic memory (few-shot past episodes)
        # 3. Raw simulation data
        user_message = (
            f"[PROCEDURAL COACHING INSTRUCTIONS]\n{instructions}\n\n"
            f"[PAST SESSION EXAMPLES ({len(past_episodes)} episodes in memory)]\n"
            f"{json.dumps(past_episodes, default=str)[:1500]}\n\n"
            f"[SIMULATION STATISTICS]\n{json.dumps(stats, indent=2, default=str)}\n\n"
            f"[IMPROVEMENT TAG FREQUENCY]\n{json.dumps(tags, indent=2)}\n\n"
            f"[PERSONALITY MATRIX]\n{json.dumps(pm, indent=2)}\n\n"
            f"[FAILURE SAMPLES (worst 20 of {len(results)} total)]\n"
            f"{json.dumps(failure_samples, indent=2, default=str)}\n\n"
            "Use the cluster_analyst subagent for deep failure cluster analysis. "
            "Also use retrieve_past_failures if you need additional context from memory. "
            "Then compile the final coaching report."
        )

        try:
            # thread_id = session_id → short-term memory continuity
            config  = {"configurable": {"thread_id": session_id}}
            context = FeedbackContext(
                user_id=user_id,
                session_id=session_id,
                twin_persona_summary=str(pm)[:200],
                total_simulations=stats.get("total", 0),
            )
            response = self._deep_agent.invoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config=config,
                context=context,
            )
            messages   = response.get("messages", [])
            final_text = messages[-1].content if messages else ""
            return self._parse_harness_output(final_text, stats, tags, pm)
        except Exception as e:
            print(f"[ClusterAnalyzer] Harness invoke error: {e}")
            return self._run_direct_llm(results, stats, tags, pm,
                                        coaching_instructions=instructions,
                                        past_episodes=past_episodes)

    # ── Fallback LangGraph graph runner ──────────────────────────

    def _run_via_graph(self, results, stats, tags, pm,
                       user_id, session_id, instructions, past_episodes) -> dict:
        initial = {
            "raw_results":           results,
            "stats":                 stats,
            "improvement_tags_freq": tags,
            "personality_matrix":    pm,
            "past_episodes":         past_episodes,
            "coaching_instructions": instructions,
            "similar_past_failures": [],
            "llm_analysis":          None,
            "final_report":          None,
        }
        config = {"configurable": {"thread_id": session_id}}
        try:
            final = self._fallback_graph.invoke(initial, config=config)
            return final.get("final_report") or self._fallback_analysis(results)
        except Exception as e:
            print(f"[ClusterAnalyzer] Graph error: {e}")
            return self._run_direct_llm(results, stats, tags, pm,
                                        coaching_instructions=instructions,
                                        past_episodes=past_episodes)

    # ── Fallback graph node functions ─────────────────────────────

    def _node_load_memory(self, state: dict) -> dict:
        """Load similar past failures from long-term semantic memory store."""
        similar = self.memory.search_similar_failures(
            "default", "failure patterns simulation", limit=3
        )
        return {"similar_past_failures": similar}

    def _node_compute_stats(self, state: dict) -> dict:
        return {"stats": self._compute_stats(state["raw_results"])}

    def _node_analyze_tags(self, state: dict) -> dict:
        return {"improvement_tags_freq": self._compute_tags(state["raw_results"])}

    def _node_personality_matrix(self, state: dict) -> dict:
        return {"personality_matrix": self._compute_personality_matrix(state["raw_results"])}

    def _node_llm_analysis(self, state: dict) -> dict:
        result = self._run_direct_llm(
            state["raw_results"],
            state["stats"],
            state["improvement_tags_freq"],
            state["personality_matrix"],
            coaching_instructions=state.get("coaching_instructions", ""),
            past_episodes=state.get("past_episodes", []),
        )
        return {"llm_analysis": result.get("llm_cluster_analysis")}

    def _node_compile_report(self, state: dict) -> dict:
        report = self._build_report(
            state["stats"],
            state["improvement_tags_freq"],
            state["personality_matrix"],
            state.get("llm_analysis"),
        )
        return {"final_report": report}

    # ── Deterministic computation (no LLM) ───────────────────────

    def _compute_stats(self, results: List[dict]) -> dict:
        by_cat: dict = defaultdict(lambda: {"total": 0, "success": 0, "partial": 0, "failure": 0})
        overall_success = 0.0
        scores = []
        for r in results:
            cat     = r.get("category", "unknown")
            verdict = r.get("verdict", "failure")
            by_cat[cat]["total"] += 1
            scores.append(r.get("overall_score", 5))
            if verdict == "success":
                by_cat[cat]["success"] += 1
                overall_success += 1
            elif verdict == "partial_success":
                by_cat[cat]["partial"] += 1
                overall_success += 0.5
            else:
                by_cat[cat]["failure"] += 1
        total = len(results)
        return {
            "total": total,
            "by_category": dict(by_cat),
            "overall_success": overall_success,
            "score_distribution": scores,
            "overall_success_rate": round(overall_success / total * 100, 1) if total else 0,
        }

    def _compute_tags(self, results: List[dict]) -> dict:
        all_tags, failed_tags = [], []
        for r in results:
            tags = r.get("improvement_tags", [])
            all_tags.extend(tags)
            if r.get("verdict") == "failure":
                failed_tags.extend(tags)
        return {
            "all":          dict(Counter(all_tags).most_common(15)),
            "failures_only": dict(Counter(failed_tags).most_common(15)),
        }

    def _compute_personality_matrix(self, results: List[dict]) -> dict:
        matrix: dict = defaultdict(lambda: {"total": 0, "success": 0})
        for r in results:
            p = r.get("counter_party_personality", "unknown")
            matrix[p]["total"] += 1
            if r.get("verdict") in ("success", "partial_success"):
                matrix[p]["success"] += 1
        return {
            p: {
                "total": v["total"],
                "success_rate": round(v["success"] / v["total"] * 100, 1) if v["total"] else 0,
            }
            for p, v in matrix.items()
        }

    # ── Direct LLM call (deepest fallback) ───────────────────────

    def _run_direct_llm(
        self,
        results: List[dict],
        stats: dict,
        tags: dict,
        pm: dict,
        coaching_instructions: str = "",
        past_episodes: Optional[List[dict]] = None,
    ) -> dict:
        if not self._gemini:
            return self._build_report(stats, tags, pm, None)

        failures = [r for r in results if r.get("verdict") == "failure"][:5]
        successes = [r for r in results if r.get("verdict") == "success"][:2]

        def _fmt_conversation(r):
            """Format a conversation into readable turn-by-turn text."""
            conv = r.get("conversation", [])
            if not conv:
                return "  (no conversation recorded)"
            lines = []
            for turn in conv:
                role = "YOUR TWIN" if turn.get("role") == "twin" else turn.get("role", "agent").upper()
                lines.append(f"  {role}: {turn.get('content', '')}")
            return "\n".join(lines)

        failure_samples = []
        for f in failures:
            failure_samples.append({
                "category":            f.get("category"),
                "counter_party":       f.get("counter_party_name"),
                "personality":         f.get("counter_party_personality"),
                "overall_score":       f.get("overall_score"),
                "tags":                f.get("improvement_tags", []),
                "key_failure_moment":  f.get("key_failure_moment"),
                "key_success_moment":  f.get("key_success_moment"),
                "alignment_reason":    f.get("alignment_reason"),
                "friction_reason":     f.get("friction_reason"),
                "outcome_reason":      f.get("outcome_reason"),
                "full_conversation":   _fmt_conversation(f),
            })

        success_samples = []
        for s in successes:
            success_samples.append({
                "category":           s.get("category"),
                "counter_party":      s.get("counter_party_name"),
                "personality":        s.get("counter_party_personality"),
                "overall_score":      s.get("overall_score"),
                "key_success_moment": s.get("key_success_moment"),
                "full_conversation":  _fmt_conversation(s),
            })

        # Inject episodic memory as few-shot context
        episode_block = ""
        if past_episodes:
            episode_block = (
                f"\n\n[PAST SESSION FEW-SHOT EXAMPLES]\n"
                f"{json.dumps(past_episodes[:2], default=str)[:1200]}"
            )

        # Inject procedural memory (coaching instructions)
        instructions_block = (
            f"\n\n[COACHING INSTRUCTIONS — follow these]\n{coaching_instructions}"
            if coaching_instructions else ""
        )

        prompt = (
            f"SIMULATION STATS:\n{json.dumps(stats, indent=2, default=str)}\n\n"
            f"TAG FREQUENCY:\n{json.dumps(tags, indent=2)}\n\n"
            f"PERSONALITY MATRIX:\n{json.dumps(pm, indent=2)}\n\n"
            f"FAILED CONVERSATIONS (with full turn-by-turn dialogue):\n"
            f"{json.dumps(failure_samples, indent=2, default=str)}\n\n"
            f"SUCCESSFUL CONVERSATIONS (for contrast):\n"
            f"{json.dumps(success_samples, indent=2, default=str)}"
            f"{episode_block}{instructions_block}\n\n"
            "Analyze every failure conversation turn by turn. Identify exactly where and why "
            "the twin's response caused the conversation to go off track. Quote the twin's actual "
            "words and explain the psychological/behavioral reason it failed. Then cluster these "
            "into failure patterns. Perform full failure cluster analysis."
        )

        try:
            response = self._gemini.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "system_instruction": CLUSTER_ANALYSIS_PROMPT,
                    "temperature": 0.4,
                    "max_output_tokens": 2048,
                    "thinking_config": {"thinking_budget": 1024},
                },
            )
            text = response.text.strip()
            for prefix in ("```json", "```", "json"):
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
            if text.endswith("```"):
                text = text[:-3].strip()
            llm_analysis = json.loads(text)
            return self._build_report(stats, tags, pm, llm_analysis)
        except Exception as e:
            print(f"[ClusterAnalyzer] Direct LLM error: {e}")
            return self._build_report(stats, tags, pm, None)

    # ── Report assembly ───────────────────────────────────────────

    def _build_report(self, stats, tags, pm, llm) -> dict:
        category_results = {}
        for cat, counts in stats.get("by_category", {}).items():
            total   = counts.get("total", 0)
            success = counts.get("success", 0)
            partial = counts.get("partial", 0)
            rate    = round((success + partial * 0.5) / total * 100, 1) if total else 0
            category_results[cat] = {
                "total": total, "successes": success,
                "partial_successes": partial,
                "failures": counts.get("failure", 0),
                "success_rate": rate,
            }
        personality_report = sorted(
            [{"personality_type": p, "success_rate": v["success_rate"], "sample_size": v["total"]}
             for p, v in pm.items()],
            key=lambda x: x["success_rate"], reverse=True,
        )
        report: dict = {
            "overall_success_rate":       stats.get("overall_success_rate", 0),
            "total_simulations":          stats.get("total", 0),
            "category_results":           category_results,
            "top_improvement_tags":       tags.get("failures_only", {}),
            "personality_success_matrix": personality_report,
            "llm_cluster_analysis":       llm or {},
        }
        if llm:
            report["failure_clusters"]   = llm.get("failure_clusters", [])
            report["critical_insight"]   = llm.get("critical_insight", "")
            report["top_improvements"]   = llm.get("top_improvements", [])
            report["success_highlights"] = llm.get("success_highlights", [])
        return report

    def _parse_harness_output(self, text: str, stats, tags, pm) -> dict:
        """Extract JSON from Deep Agents harness output."""
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return self._build_report(stats, tags, pm, json.loads(match.group()))
            except json.JSONDecodeError:
                pass
        return self._build_report(stats, tags, pm, None)

    def _fallback_analysis(self, results: List[dict]) -> dict:
        stats = self._compute_stats(results)
        tags  = self._compute_tags(results)
        pm    = self._compute_personality_matrix(results)
        report = self._build_report(stats, tags, pm, None)
        report["critical_insight"] = "Enable Gemini API for deep cluster analysis."
        return report
