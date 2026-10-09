"""
LiveTwinSessionManager
========================
Orchestrates the full Live Digital Twin flow, in-memory (mirrors the
`jobs` / `_simulations` dict pattern used elsewhere in this app):

  1. describe   → user describes the person they'll face
  2. questions  → LiveTwinQuestionGenerator asks 5-8 follow-ups
  3. build      → LiveTwinPersonaBuilder embodies that person
  4. live_chat  → user talks live (webcam+mic); LiveTwinChatAgent replies;
                  LiveBehaviorTracker ingests frames/audio in the background
  5. judged     → LivePerformanceJudge grades the real performance
"""

import threading
import uuid
from datetime import datetime
from typing import Optional

from agent.live.live_question_generator import LiveTwinQuestionGenerator
from agent.live.live_persona_builder import LiveTwinPersonaBuilder
from agent.live.live_twin_chat import LiveTwinChatAgent
from agent.live.live_behavior_tracker import LiveBehaviorTracker
from agent.live.live_judge import LivePerformanceJudge
from agent.live.use_cases import get_use_case
from agent.live.streak_manager import get_streak_manager
from agent.token_logger import run_context, track_stage, log_run_summary

_sessions: dict = {}

# A single session_id becomes the "run_id" for every LLM call and analysis
# step across its whole lifetime (spans many separate HTTP requests), so
# /admin/runs/<session_id>/telemetry and /admin/token-usage can show exactly
# which step of a given live session used how many tokens and when.
_RUN_TYPE = "live_twin_session"


class LiveTwinSessionManager:
    def __init__(self):
        self._question_gen = LiveTwinQuestionGenerator()
        self._persona_builder = LiveTwinPersonaBuilder()
        self._chat_agent = LiveTwinChatAgent()
        self._judge = LivePerformanceJudge()
        self._streaks = get_streak_manager()

    # ── Phase 1-2: describe → questions ────────────────────────────
    def start_description(
        self,
        description: str,
        user_id: Optional[str] = None,
        use_case_id: Optional[str] = None,
    ) -> dict:
        session_id = uuid.uuid4().hex[:12]
        use_case = get_use_case(use_case_id) if use_case_id else None

        # Current difficulty level for this user+use-case, from their Vyakti Streak.
        level = 1
        if user_id and use_case_id:
            level = self._streaks.get_status(user_id, use_case_id).get("level", 1)

        with run_context(session_id, _RUN_TYPE, user_id or ""):
            questions = self._question_gen.generate_questions(description, user_id=user_id, use_case=use_case)
        _sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "use_case_id": use_case_id,
            "level": level,
            "description": description,
            "questions": questions,
            "answers": {},
            "persona": None,
            "conversation": [],
            "tracker": LiveBehaviorTracker(),
            "phase": "awaiting_answers",
            "judgment": None,
            "created_at": datetime.utcnow().isoformat(),
            # Per-session background frame-analysis worker bookkeeping — see
            # ingest_frame() for why this exists.
            "_frame_lock": threading.Lock(),
            "_pending_frame": None,
            "_frame_worker_running": False,
        }
        return {"session_id": session_id, "questions": questions, "use_case_id": use_case_id, "level": level}

    # ── Phase 3: build persona ──────────────────────────────────────
    def build_twin(self, session_id: str, answers: dict) -> dict:
        session = self._require(session_id)
        session["answers"] = answers or {}
        use_case = get_use_case(session.get("use_case_id")) if session.get("use_case_id") else None
        with run_context(session_id, _RUN_TYPE, session["user_id"] or ""):
            persona = self._persona_builder.build(
                session["description"],
                session["answers"],
                user_id=session["user_id"],
                use_case=use_case,
                level=session.get("level", 1),
            )
        session["persona"] = persona
        session["phase"] = "live_chat"

        opening_line = persona.get("opening_line", "Hi — let's get started.")
        session["conversation"].append({"role": "twin", "content": opening_line})

        return {
            "session_id": session_id,
            "persona": {k: v for k, v in persona.items() if not k.startswith("_")},
            "opening_line": opening_line,
            "level": session.get("level", 1),
        }

    # ── Phase 4: live conversation ───────────────────────────────────
    def send_message(self, session_id: str, message: str) -> dict:
        session = self._require(session_id)
        if not session.get("persona"):
            return {"error": "Twin has not been built yet"}

        session["conversation"].append({"role": "user", "content": message})
        session["tracker"].mark_turn("user", message)
        behavior_hint = session["tracker"].get_behavior_hint()
        with run_context(session_id, _RUN_TYPE, session["user_id"] or ""):
            reply = self._chat_agent.respond(
                persona=session["persona"],
                conversation=session["conversation"],
                user_message=message,
                behavior_hint=behavior_hint or None,
                user_id=session["user_id"],
            )
        session["conversation"].append({"role": "twin", "content": reply})
        session["tracker"].mark_turn("twin", reply)
        return {"reply": reply}

    def ingest_frame(self, session_id: str, image_b64: str) -> bool:
        # A single frame's full analysis (DeepFace + 3x MediaPipe Tasks, run
        # sequentially) can take far longer than the ~2.5s the browser waits
        # between captures on this hardware (measured 10-200+s under load).
        # Running it synchronously blocked a shared gunicorn worker thread for
        # that whole duration, starving /live/message until it timed out
        # client-side. Process in a dedicated background thread per session
        # instead, keeping only the LATEST pending frame and dropping any that
        # arrive while one is already being processed — for behavioral
        # averaging a recent sample is enough; queuing every frame would only
        # grow a backlog this hardware can never catch up on.
        session = self._require(session_id)
        lock = session["_frame_lock"]
        with lock:
            session["_pending_frame"] = image_b64
            if session["_frame_worker_running"]:
                return True
            session["_frame_worker_running"] = True

        def _drain():
            while True:
                with lock:
                    frame = session["_pending_frame"]
                    session["_pending_frame"] = None
                    if frame is None:
                        session["_frame_worker_running"] = False
                        return
                try:
                    with track_stage(
                        "live_frame_analysis", "DeepFace+MediaPipe(Pose/Hand/Face)", "non_llm",
                        extra={"run_id": session_id, "run_type": _RUN_TYPE, "user_id": session["user_id"] or ""},
                    ):
                        session["tracker"].ingest_frame_b64(frame)
                except Exception as e:
                    print(f"[LiveTwinSessionManager] Background frame analysis failed: {e}")

        threading.Thread(target=_drain, daemon=True).start()
        return True

    def ingest_audio(self, session_id: str, audio_b64: str, suffix: str = ".webm") -> Optional[dict]:
        # Same rationale as ingest_frame: don't block the shared worker-thread
        # pool on CPU-heavy analysis. Audio chunks are smaller/cheaper than
        # frames and matter more for transcript completeness, so every chunk
        # gets processed (no dropping) — just off the request-handling thread.
        session = self._require(session_id)

        def _process():
            try:
                with track_stage(
                    "live_audio_analysis", "Librosa+SpeechRecognition", "non_llm",
                    extra={"run_id": session_id, "run_type": _RUN_TYPE, "user_id": session["user_id"] or ""},
                ):
                    session["tracker"].ingest_audio_b64(audio_b64, suffix=suffix)
            except Exception as e:
                print(f"[LiveTwinSessionManager] Background audio analysis failed: {e}")

        threading.Thread(target=_process, daemon=True).start()
        return {"ok": True}

    # ── Phase 5: judge performance ───────────────────────────────────
    def end_session(self, session_id: str) -> dict:
        session = self._require(session_id)
        tracker = session["tracker"]
        behavior_summary = tracker.get_aggregated_summary()
        turn_segments = tracker.get_turn_segments()
        use_case = get_use_case(session.get("use_case_id")) if session.get("use_case_id") else None

        with run_context(session_id, _RUN_TYPE, session["user_id"] or ""):
            judgment = self._judge.judge(
                situation_description=session["description"],
                persona=session["persona"] or {},
                conversation=session["conversation"],
                behavior_summary=behavior_summary,
                turn_segments=turn_segments,
                use_case=use_case,
                user_id=session["user_id"],
            )

        # Vyakti Streak: record this session against the chosen use case, if any.
        if session.get("user_id") and session.get("use_case_id"):
            judgment["streak"] = self._streaks.record_session(
                session["user_id"], session["use_case_id"], judgment
            )

        session["judgment"] = judgment
        session["phase"] = "judged"
        tracker.release()

        # Aggregated token/cost/duration summary for this entire run, printed to
        # the console and appended to logs/token_usage.jsonl as a run_summary line.
        log_run_summary(
            run_type=_RUN_TYPE,
            sim_id=session_id,
            user_id=session.get("user_id"),
            started_at=session.get("created_at"),
        )
        return judgment

    # ── status ────────────────────────────────────────────────────
    def get_session(self, session_id: str) -> Optional[dict]:
        session = _sessions.get(session_id)
        if not session:
            return None
        return {
            "session_id": session_id,
            "phase": session["phase"],
            "use_case_id": session.get("use_case_id"),
            "level": session.get("level"),
            "description": session["description"],
            "persona": {k: v for k, v in (session["persona"] or {}).items() if not k.startswith("_")},
            "conversation": session["conversation"],
            "judgment": session["judgment"],
        }

    def _require(self, session_id: str) -> dict:
        session = _sessions.get(session_id)
        if not session:
            raise KeyError(f"Live twin session not found: {session_id}")
        return session


_manager: Optional[LiveTwinSessionManager] = None


def get_live_session_manager() -> LiveTwinSessionManager:
    global _manager
    if _manager is None:
        _manager = LiveTwinSessionManager()
    return _manager
