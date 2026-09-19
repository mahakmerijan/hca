"""
LivePerformanceJudge
=====================
Final step of the Live Digital Twin flow: a neutral Gemini judge grades the
user's REAL live performance during the conversation, combining:
  - the situation they described (what they're preparing for)
  - the persona they were speaking with
  - the full conversation transcript
  - quantitative behavioral signals captured live from their webcam/mic:
      * voice tone / pace / energy — and whether it fit the moment
      * posture & body openness (MediaPipe Pose)
      * hand movement / gesture activity (MediaPipe Hands)
      * facial expression / emotion distribution (DeepFace)
      * eyebrow movement (MediaPipe FaceLandmarker blendshapes)
      * speech content quality (filler words, clarity, vocabulary)

The prompt is built dynamically from these live-measured numbers so the
LLM's qualitative judgement is grounded in what was actually measured,
not just the transcript.
"""

import json
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

_SYSTEM_PROMPT = """You are a neutral performance judge for a live conversation-practice session.
A user just had a REAL, live video conversation with an AI roleplaying a specific person they
described (e.g. a recruiter, investor, date, boss). You are given the situation, who they were
talking to, the full transcript, measured behavioral signals captured live from their webcam and
microphone during the call, and a turn-by-turn breakdown of those signals. Judge the user's actual
performance.

Score each dimension 1-10 and ground every score/reason in the transcript AND the measured
signals given to you — do not invent details that aren't supported by the data.

For "moment_by_moment": walk through the TURN-BY-TURN SIGNALS you were given and call out the
specific points where the user's facial expression, eyebrow movement, or posture worked against them
(or, when there's nothing bad to say about a strong turn, note what worked). Quote their exact words
for each moment so it's traceable to the transcript. Produce one entry per notable turn (aim for as
many as there are meaningful turns, at least 3 if the conversation is long enough).

For "growth_hexagon": score these 6 axes 1-10, used to track the user's long-term progress:
  - composure_under_pressure: how well they held together under pushback/hostile moments
  - vocal_resonance_and_control: tone stability and pacing
  - posture_and_openness: shoulder alignment, lack of defensive/crossed-arm body language
  - facial_congruence: whether facial expressions matched the gravity/tone of the topic
  - conciseness_and_clarity: eliminating rambling and filler words
  - active_listening_markers: strategic pauses before answering, responsiveness to what was said

For "confidence_score": a single 0-100 composite score summarizing overall performance strength.

Return ONLY valid JSON with this schema:
{
  "overall_score": <1-10>,
  "verdict": "strong | solid | needs_work | poor",
  "situation_fit_summary": "1-2 sentences: did their approach fit the specific situation described?",
  "dimensions": {
    "voice_and_tone": {"score": <1-10>, "reason": "was their tone/pace/energy appropriate to what was being asked/discussed at each moment?"},
    "posture_and_body_language": {"score": <1-10>, "reason": "openness, shoulder/spine alignment, lean direction"},
    "facial_expressions": {"score": <1-10>, "reason": "emotional appropriateness, smiling vs neutral vs tense, matching the conversation's mood"},
    "eyebrow_and_micro_expressions": {"score": <1-10>, "reason": "engagement/reactivity signaled by eyebrow movement — frozen face vs over-animated vs naturally expressive"},
    "confidence": {"score": <1-10>, "reason": "based on posture confidence signals, pace, filler words, arms-crossed ratio"},
    "talking_quality": {"score": <1-10>, "reason": "clarity, structure, filler words, vocabulary, directness of answers in the transcript"},
    "hand_movements": {"score": <1-10>, "reason": "gesture activity level — too stiff, too fidgety, or natural and purposeful"}
  },
  "moment_by_moment": [
    {
      "quote": "exact user line from transcript",
      "facial_expression_issue": "what was wrong with facial expression here, or 'appropriate' if fine",
      "eyebrow_issue": "what was wrong with eyebrow/engagement signals here, or 'appropriate' if fine",
      "posture_issue": "what was wrong with posture/openness here, or 'appropriate' if fine",
      "how_to_improve": "specific, actionable fix for this exact moment"
    }
  ],
  "growth_hexagon": {
    "composure_under_pressure": <1-10>,
    "vocal_resonance_and_control": <1-10>,
    "posture_and_openness": <1-10>,
    "facial_congruence": <1-10>,
    "conciseness_and_clarity": <1-10>,
    "active_listening_markers": <1-10>
  },
  "confidence_score": <0-100>,
  "strengths": ["...", "...", "..."],
  "areas_to_improve": ["...", "...", "..."],
  "best_moment": {"quote": "exact line from transcript", "why": "..."},
  "worst_moment": {"quote": "exact line from transcript", "why": "..."},
  "coaching_tips": ["specific, actionable tip", "...", "..."]
}
Verdicts: strong >= 8, solid >= 6, needs_work >= 4, poor < 4 (based on overall_score).
No preamble, no markdown — pure JSON object."""


class LivePerformanceJudge:
    def __init__(self):
        # Same model used for question generation / persona building, for consistent behavior.
        self.model_name = os.getenv("LLM_MODEL", "gemini-2.5-pro")
        self._project = os.getenv("VERTEX_PROJECT", "ai-ml-integrations")
        self._location = os.getenv("VERTEX_LOCATION", "us-central1")
        self.available = GENAI_AVAILABLE
        if self.available:
            self.client = genai.Client(vertexai=True, project=self._project, location=self._location)

    def judge(
        self,
        situation_description: str,
        persona: dict,
        conversation: list,
        behavior_summary: dict,
        turn_segments: Optional[list] = None,
        use_case: Optional[dict] = None,
        user_id: Optional[str] = None,
    ) -> dict:
        if not self.available:
            return self._fallback_judgment()

        conversation_text = "\n".join(
            f"{('USER' if t['role']=='user' else persona.get('persona_name','TWIN').upper())}: {t['content']}"
            for t in conversation
        )

        fa = behavior_summary.get("facial_expression", {})
        bl = behavior_summary.get("body_language", {})
        brow = behavior_summary.get("eyebrow_and_face", {})
        voice = behavior_summary.get("voice", {})
        bl_metrics = bl.get("average_metrics", {})

        signals_text = (
            f"FACIAL EXPRESSION — frames analyzed: {fa.get('total_frames_analyzed', 0)}, "
            f"smile ratio: {fa.get('smile_ratio', 'n/a')}, "
            f"emotion distribution: {fa.get('emotion_distribution', {})}\n\n"
            f"POSTURE / BODY LANGUAGE — pose frames detected: {bl.get('pose_detected_count', 0)}, "
            f"avg shoulder alignment: {bl_metrics.get('avg_shoulder_alignment', 'n/a')}, "
            f"avg head uprightness: {bl_metrics.get('avg_head_uprightness', 'n/a')}, "
            f"avg openness: {bl_metrics.get('avg_openness', 'n/a')}, "
            f"avg confidence score: {bl_metrics.get('avg_confidence_score', 'n/a')}, "
            f"arms crossed ratio: {bl_metrics.get('arms_crossed_ratio', 'n/a')}, "
            f"scenario gesture/hand-movement scores: {bl.get('scenario_scores', {})}\n\n"
            f"EYEBROW / FACE BLENDSHAPES — movement activity: {brow.get('eyebrow_movement_activity', 'n/a')}, "
            f"raise ratio: {brow.get('eyebrow_raise_ratio', 'n/a')}, "
            f"furrow ratio: {brow.get('eyebrow_furrow_ratio', 'n/a')}\n\n"
            f"VOICE — chunks analyzed: {voice.get('chunks_analyzed', 0)}, "
            f"avg pitch: {voice.get('avg_pitch', 'n/a')}, avg energy: {voice.get('avg_energy', 'n/a')}, "
            f"dominant pace: {voice.get('dominant_pace', 'n/a')}, "
            f"avg pause ratio: {voice.get('avg_pause_ratio', 'n/a')}, "
            f"avg filler word ratio: {voice.get('avg_filler_word_ratio', 'n/a')}"
        )

        turn_segments_text = ""
        if turn_segments:
            lines = []
            for seg in turn_segments:
                lines.append(
                    f"Turn {seg['turn_index']} — user said: \"{seg['user_said']}\" | "
                    f"dominant_emotion={seg.get('dominant_emotion')}, "
                    f"confidence_score={seg.get('avg_confidence_score')}, "
                    f"openness={seg.get('avg_openness')}, "
                    f"arms_crossed_ratio={seg.get('arms_crossed_ratio')}, "
                    f"eyebrow_raise={seg.get('avg_eyebrow_raise')}, "
                    f"eyebrow_furrow={seg.get('avg_eyebrow_furrow')}"
                )
            turn_segments_text = "\n\nTURN-BY-TURN SIGNALS (facial/posture/eyebrow measured during each of the user's turns):\n" + "\n".join(lines)

        use_case_block = f"\n\nDOMAIN-SPECIFIC JUDGING FOCUS: {use_case['judge_focus']}" if use_case else ""

        user_prompt = (
            f"SITUATION THE USER DESCRIBED:\n{situation_description}\n\n"
            f"WHO THEY WERE TALKING TO (AI persona):\n"
            f"{persona.get('persona_name','')} — {persona.get('persona_role','')}. "
            f"{persona.get('personality_summary','')}\n\n"
            f"FULL TRANSCRIPT:\n{conversation_text}\n\n"
            f"MEASURED LIVE BEHAVIORAL SIGNALS (from webcam/mic during the call):\n{signals_text}"
            f"{turn_segments_text}"
            f"{use_case_block}\n\n"
            "Judge the user's live performance now."
        )

        try:
            from agent.token_logger import extract_token_counts, log_call
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config={
                    "system_instruction": _SYSTEM_PROMPT,
                    "temperature": 0.3,
                    "max_output_tokens": 8192,
                    # Minimal thinking budget — gemini-2.5-pro rejects 0, and unbounded
                    # thinking was silently eating the output budget, truncating the JSON.
                    "thinking_config": {"thinking_budget": 128},
                },
            )
            input_tokens, output_tokens = extract_token_counts(response)
            log_call(self.model_name, "LivePerformanceJudge", input_tokens, output_tokens, user_id=user_id)

            raw = response.text
            if raw is None:
                try:
                    raw = response.candidates[0].content.parts[0].text or ""
                except Exception:
                    raw = ""
            text = raw.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = text[:-3].strip()
            if text.startswith("json"):
                text = text[4:].strip()

            import re
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                text = m.group(0)

            judgment = json.loads(text)
            judgment["behavior_summary"] = behavior_summary
            judgment["turn_segments"] = turn_segments or []
            self._ensure_computed_fields(judgment)
            return judgment
        except Exception as e:
            print(f"[LivePerformanceJudge] Error: {e}")
            fb = self._fallback_judgment()
            fb["behavior_summary"] = behavior_summary
            fb["turn_segments"] = turn_segments or []
            return fb

    def _ensure_computed_fields(self, judgment: dict):
        """Defensive fallback if the model omits confidence_score/growth_hexagon."""
        dims = judgment.get("dimensions", {}) or {}
        dim_scores = [d.get("score", 5) for d in dims.values() if isinstance(d, dict)]
        avg_dim = (sum(dim_scores) / len(dim_scores)) if dim_scores else 5

        if not isinstance(judgment.get("confidence_score"), (int, float)):
            judgment["confidence_score"] = round(avg_dim * 10, 1)

        if not isinstance(judgment.get("growth_hexagon"), dict) or not judgment["growth_hexagon"]:
            judgment["growth_hexagon"] = {
                "composure_under_pressure": dims.get("confidence", {}).get("score", 5),
                "vocal_resonance_and_control": dims.get("voice_and_tone", {}).get("score", 5),
                "posture_and_openness": dims.get("posture_and_body_language", {}).get("score", 5),
                "facial_congruence": dims.get("facial_expressions", {}).get("score", 5),
                "conciseness_and_clarity": dims.get("talking_quality", {}).get("score", 5),
                "active_listening_markers": dims.get("eyebrow_and_micro_expressions", {}).get("score", 5),
            }

        if not isinstance(judgment.get("moment_by_moment"), list):
            judgment["moment_by_moment"] = []

    def _fallback_judgment(self) -> dict:
        return {
            "overall_score": 5,
            "verdict": "needs_work",
            "situation_fit_summary": "Unable to grade — Gemini unavailable.",
            "dimensions": {
                k: {"score": 5, "reason": "Unable to grade"}
                for k in ("voice_and_tone", "posture_and_body_language", "facial_expressions",
                          "eyebrow_and_micro_expressions", "confidence", "talking_quality", "hand_movements")
            },
            "moment_by_moment": [],
            "growth_hexagon": {
                "composure_under_pressure": 5, "vocal_resonance_and_control": 5,
                "posture_and_openness": 5, "facial_congruence": 5,
                "conciseness_and_clarity": 5, "active_listening_markers": 5,
            },
            "confidence_score": 50,
            "strengths": [],
            "areas_to_improve": [],
            "best_moment": None,
            "worst_moment": None,
            "coaching_tips": [],
        }
