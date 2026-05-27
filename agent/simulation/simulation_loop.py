"""
Simulation Loop
===============
LangGraph-based multi-agent orchestration engine.
Runs the Social Ping loop (4-step micro-simulation) for each scenario.

Steps per scenario (architecture.md Step 2):
  1. Opening   — Twin delivers pitch/intro
  2. Reaction  — Counter-party responds
  3. Curveball — Counter-party throws a hard question
  4. Closing   — Twin attempts to seal the deal

Uses Gemini context caching for efficiency during the 10-run batch.
"""

import json
import os
import uuid
import threading
from typing import Dict, List, Optional, Callable
from dotenv import load_dotenv

load_dotenv()

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from typing import TypedDict, Annotated
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from .counter_agents import get_agent_for_category
from .referee import RefereeAgent


_GENAI_CLIENT = None
_GENAI_CLIENT_LOCK = threading.Lock()


def _get_genai_client():
    global _GENAI_CLIENT
    if _GENAI_CLIENT is not None:
        return _GENAI_CLIENT
    with _GENAI_CLIENT_LOCK:
        if _GENAI_CLIENT is None:
            _GENAI_CLIENT = genai.Client(
                vertexai=True,
                project=os.getenv("VERTEX_PROJECT", "ai-ml-integrations"),
                location=os.getenv("VERTEX_LOCATION", "us-central1"),
            )
    return _GENAI_CLIENT


# ── LangGraph State Schema ────────────────────────────────────────

if LANGGRAPH_AVAILABLE:
    from typing import TypedDict, Annotated
    from langgraph.graph.message import add_messages

    class SimulationState(TypedDict):
        scenario: dict
        twin_persona: dict
        conversation: Annotated[list, add_messages]
        stage: str
        grade: Optional[dict]
        completed: bool


def _build_twin_response(twin_persona: dict, context: str, stage: str) -> str:
    """
    Generate the Digital Twin's response using the persona's system prompt
    and the cached Gemini context (if available).
    """
    if not GENAI_AVAILABLE:
        return "[Twin response — Gemini unavailable]"

    # Use flash model for simulation turns — gemini-2.5-pro thinking tokens
    # consume max_output_tokens, leaving nothing for the actual response.
    model_name = os.getenv("SIM_LLM_MODEL", "gemini-2.0-flash")
    client = _get_genai_client()
    system = twin_persona.get("system_prompt") or "You are a digital twin. Respond authentically in 2-4 sentences."
    if not system or len(system) < 10:
        print(f"[SimLoop] WARNING: twin_persona has no/empty system_prompt. Keys: {list(twin_persona.keys())}")
    user = f"Stage: {stage}\nSituation: {context}\n\nRespond authentically as yourself (2-4 sentences max):"

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=user,
            config={
                "system_instruction": system,
                "temperature": 0.4,
                "max_output_tokens": 300,
            },
        )
        text = response.text
        if text is None:
            # Log the actual finish reason to diagnose why Gemini returns None
            try:
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, 'finish_reason', 'unknown')
                print(f"[SimLoop] Gemini returned None text. finish_reason={finish_reason}")
                # Try extracting from parts directly
                text = candidate.content.parts[0].text
            except Exception as ex:
                print(f"[SimLoop] Could not extract text from candidate: {ex}")
                text = ""
        return (text or "").strip() or "[Twin is composing a response...]"
    except Exception as e:
        return f"[Twin error: {e}]"


class SimulationLoop:
    """
    Orchestrates the full 10-scenario simulation run using LangGraph.
    Each scenario runs a 4-step Social Ping micro-simulation.
    """

    def __init__(self, twin_persona: Optional[dict] = None, gemini_cache_name: Optional[str] = None):
        self.twin_persona = twin_persona or {}
        self.gemini_cache_name = gemini_cache_name
        self.referee = RefereeAgent()
        self._graph = self._build_graph() if LANGGRAPH_AVAILABLE else None

    # ── LangGraph graph construction ──────────────────────────────

    def _build_graph(self):
        """Build the LangGraph state machine for one micro-simulation."""
        if not LANGGRAPH_AVAILABLE:
            return None

        graph = StateGraph(SimulationState)

        graph.add_node("opening", self._node_opening)
        graph.add_node("reaction", self._node_reaction)
        graph.add_node("curveball", self._node_curveball)
        graph.add_node("closing", self._node_closing)
        graph.add_node("grade", self._node_grade)

        graph.set_entry_point("opening")
        graph.add_edge("opening", "reaction")
        graph.add_edge("reaction", "curveball")
        graph.add_edge("curveball", "closing")
        graph.add_edge("closing", "grade")
        graph.add_edge("grade", END)

        return graph.compile()

    # ── Node functions ────────────────────────────────────────────

    def _node_opening(self, state: dict) -> dict:
        scenario = state["scenario"]
        opening_ctx = scenario.get("opening_prompt", "Introduce yourself.")
        twin_resp = _build_twin_response(
            self.twin_persona,
            context=opening_ctx,
            stage="opening",
        )
        agent = get_agent_for_category(scenario["category"])
        agent_resp = agent.respond(scenario["counter_party"], twin_resp, stage="opening")

        return {
            "conversation": [
                {"role": "twin", "content": twin_resp},
                {"role": "agent", "content": agent_resp},
            ],
            "stage": "reaction",
        }

    def _node_reaction(self, state: dict) -> dict:
        conv = state["conversation"]
        last_agent = conv[-1]["content"] if conv else ""
        scenario = state["scenario"]

        twin_resp = _build_twin_response(
            self.twin_persona,
            context=f"They said: {last_agent}",
            stage="reaction",
        )
        agent = get_agent_for_category(scenario["category"])
        agent_resp = agent.respond(scenario["counter_party"], twin_resp, stage="reaction")

        return {
            "conversation": [
                {"role": "twin", "content": twin_resp},
                {"role": "agent", "content": agent_resp},
            ],
            "stage": "curveball",
        }

    def _node_curveball(self, state: dict) -> dict:
        scenario = state["scenario"]
        curveball = scenario.get("curveball_question", "Tell me more.")

        twin_resp = _build_twin_response(
            self.twin_persona,
            context=f"Curveball question: '{curveball}'",
            stage="curveball",
        )
        agent = get_agent_for_category(scenario["category"])
        agent_resp = agent.respond(scenario["counter_party"], twin_resp, stage="curveball")

        return {
            "conversation": [
                {"role": "agent", "content": curveball},  # agent asks curveball
                {"role": "twin", "content": twin_resp},
                {"role": "agent", "content": agent_resp},
            ],
            "stage": "closing",
        }

    def _node_closing(self, state: dict) -> dict:
        scenario = state["scenario"]
        closing_ctx = scenario.get("closing_prompt", "Final impression?")
        conv = state["conversation"]
        last_agent = conv[-1]["content"] if conv else ""

        twin_resp = _build_twin_response(
            self.twin_persona,
            context=f"Closing: {closing_ctx}. They last said: {last_agent}",
            stage="closing",
        )
        return {
            "conversation": [{"role": "twin", "content": twin_resp}],
            "stage": "grade",
        }

    def _node_grade(self, state: dict) -> dict:
        grade = self.referee.grade(
            scenario=state["scenario"],
            conversation=state["conversation"],
            twin_persona=self.twin_persona,
        )
        return {"grade": grade, "completed": True}

    # ── Public runner ─────────────────────────────────────────────

    def run_single(self, scenario: dict, max_turns: int = 50) -> dict:
        """Run one full multi-turn simulation up to max_turns dialogue exchanges."""
        return self._extended_run(scenario, max_turns=max_turns)

    def _extended_run(self, scenario: dict, max_turns: int = 50) -> dict:
        """
        Extended multi-turn conversation loop.
        Runs up to max_turns back-and-forth exchanges between the twin and counter-party.
        Stages progress naturally: opening → building → curveball → pressure → closing.
        """
        conversation = []
        agent = get_agent_for_category(scenario["category"])
        total_turns = max_turns  # each turn = 1 twin + 1 agent message

        # Stage thresholds (as fraction of total turns)
        def _stage_for_turn(t):
            pct = t / total_turns
            if pct < 0.15:   return "opening"
            elif pct < 0.45: return "reaction"
            elif pct < 0.65: return "curveball"
            elif pct < 0.85: return "pressure"
            else:            return "closing"

        curveball_used = False
        closing_context = scenario.get("closing_prompt", "Wrap up the conversation.")

        for turn_idx in range(total_turns):
            stage = _stage_for_turn(turn_idx)

            # Build context for the twin from the last agent message
            if not conversation:
                ctx = scenario.get("opening_prompt", "Introduce yourself and begin the conversation.")
            elif stage == "curveball" and not curveball_used:
                curveball_used = True
                ctx = f"Curveball: '{scenario.get('curveball_question', 'Tell me something unexpected.')}'"
            elif stage == "closing" and turn_idx == total_turns - 1:
                ctx = f"Final closing — {closing_context}"
            else:
                last_agent_msg = next(
                    (m["content"] for m in reversed(conversation) if m["role"] == "agent"), ""
                )
                ctx = f"They said: \"{last_agent_msg}\""

            # Twin responds
            twin_resp = _build_twin_response(self.twin_persona, context=ctx, stage=stage)
            conversation.append({"role": "twin", "content": twin_resp})

            # Agent responds (except on the very last turn)
            if turn_idx < total_turns - 1:
                agent_resp = agent.respond(scenario["counter_party"], twin_resp, stage=stage)
                conversation.append({"role": "agent", "content": agent_resp})

        grade = self.referee.grade(scenario, conversation, self.twin_persona)
        grade["conversation"] = conversation
        return grade

    def _simple_run(self, scenario: dict) -> dict:
        """Simplified simulation without LangGraph (fallback) — also runs extended loop."""
        return self._extended_run(scenario, max_turns=50)

    def run_batch(
        self,
        scenarios: List[dict],
        twin_persona: Optional[dict] = None,
        progress_callback: Optional[Callable[[int, int, dict], None]] = None,
        max_workers: int = 5,
    ) -> List[dict]:
        """
        Run all scenarios. Uses threading for concurrent execution.
        progress_callback(completed, total, last_result) is called after each scenario.
        """
        if twin_persona:
            self.twin_persona = twin_persona
        results = []
        lock = threading.Lock()
        completed = [0]

        def run_one(scenario):
            result = self.run_single(scenario)
            result["scenario_id"] = scenario.get("scenario_id")
            result["category"] = scenario.get("category", "")
            result["counter_party_name"] = scenario.get("counter_party", {}).get("name", "Agent")
            with lock:
                results.append(result)
                completed[0] += 1
                if progress_callback:
                    progress_callback(completed[0], len(scenarios), result)

        # Use thread pool for concurrency (respects Gemini rate limits via max_workers)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(run_one, scenarios)

        return results
