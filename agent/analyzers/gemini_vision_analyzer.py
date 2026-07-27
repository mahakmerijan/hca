"""
Gemini Vision Analyzer
======================
Replaces MediaPipe (body language) + DeepFace (facial expression) with a
single batched Gemini Vision API call across all sampled video frames.

One API call covers ALL frames — minimal cost, zero native ML libraries.
Memory: ~30–80 KB per buffered JPEG vs ~200 MB for MediaPipe models.

Token usage is logged to logs/token_usage.jsonl.
"""

import json
import logging
import os
import re

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    genai = None
    genai_types = None
    _GENAI_AVAILABLE = False

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a video behaviour analyst. Analyse each frame precisely and objectively. "
    "Return ONLY valid JSON — no markdown fences, no explanation, no preamble."
)

_USER_TMPL = """\
Below are {n} sampled video frames labelled Frame 0 … Frame {last}.
For EACH frame return ONE JSON object inside a top-level array.
Use these exact frame_idx values in order: {indices}

Each object must have EXACTLY this structure:

{{
  "frame_idx": <integer>,
  "face_detected": <true|false>,
  "dominant_emotion": "<happy|sad|angry|fear|surprise|disgust|neutral>",
  "emotions": {{
    "happy": <0.0–1.0>,
    "sad": <0.0–1.0>,
    "angry": <0.0–1.0>,
    "fear": <0.0–1.0>,
    "surprise": <0.0–1.0>,
    "disgust": <0.0–1.0>,
    "neutral": <0.0–1.0>
  }},
  "smile_detected": <true|false>,
  "pose_detected": <true|false>,
  "posture": {{
    "shoulder_alignment": <0.0–1.0, 1=perfectly level>,
    "head_uprightness": <0.0–1.0, 1=perfectly upright>,
    "spine_straightness": <0.0–1.0>,
    "openness": <0.0–1.0, 1=very open stance>
  }},
  "confidence_signals": {{
    "arms_crossed": <true|false>,
    "lean_direction": "<left|right|centered>",
    "forward_lean": <0.0–1.0>,
    "confidence_score": <0.0–1.0>
  }}
}}

Return a JSON ARRAY of exactly {n} objects in the same order as the frames.
"""

# ── Emotion → scenario impact weights (mirrors facial_expression.py) ─────────
_SCENARIO_IMPACT = {
    "job_interview": {"happy": 0.85, "neutral": 0.70, "surprise": 0.40,
                      "sad": 0.10, "angry": 0.05, "fear": 0.15, "disgust": 0.05},
    "business_deal": {"happy": 0.80, "neutral": 0.75, "surprise": 0.35,
                      "sad": 0.10, "angry": 0.10, "fear": 0.10, "disgust": 0.05},
    "date":          {"happy": 0.95, "surprise": 0.60, "neutral": 0.40,
                      "sad": 0.10, "angry": 0.02, "fear": 0.10, "disgust": 0.02},
}


class GeminiVisionAnalyzer:
    """
    Buffers sampled video frames (as JPEG bytes) then runs ONE batched
    Gemini Vision call to analyse facial expressions AND body language.

    Interface is compatible with the old FacialExpressionAnalyzer +
    BodyLanguageAnalyzer pair — use get_facial_summary() and get_body_summary()
    where the old get_summary() calls were.
    """

    def __init__(self, job_id: str = ""):
        self.job_id = job_id
        self.model_name = os.getenv("LLM_MODEL", "gemini-2.5-flash")
        self._frame_buffer: list[tuple[int, bytes]] = []  # (frame_idx, jpeg_bytes)
        self._results: list[dict] = []
        self._analysed = False
        self._client = None

        if _GENAI_AVAILABLE:
            try:
                self._client = genai.Client(
                    vertexai=True,
                    project=os.getenv("VERTEX_PROJECT", "ai-ml-integrations"),
                    location=os.getenv("VERTEX_LOCATION", "us-central1"),
                )
            except Exception as exc:
                logger.warning("[GeminiVisionAnalyzer] Client init failed: %s", exc)

    # ── Frame buffering ───────────────────────────────────────────────────────

    def analyze_frame(self, frame: np.ndarray, frame_idx: int):
        """
        Encode and buffer one frame. The actual Gemini call is deferred
        until get_facial_summary() or get_body_summary() is called.
        """
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640.0 / w
            frame = cv2.resize(frame, (640, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            self._frame_buffer.append((frame_idx, buf.tobytes()))

    # ── Gemini API call ───────────────────────────────────────────────────────

    def _run_analysis(self):
        """Make the single batched Gemini Vision call (idempotent)."""
        if self._analysed:
            return
        self._analysed = True

        if not self._frame_buffer:
            logger.warning("[GeminiVisionAnalyzer] No frames buffered.")
            return

        indices = [idx for idx, _ in self._frame_buffer]
        n = len(indices)

        if self._client is None or not _GENAI_AVAILABLE:
            logger.warning("[GeminiVisionAnalyzer] No Gemini client — using fallbacks.")
            self._results = [_fallback_frame(idx) for idx in indices]
            return

        prompt_text = _USER_TMPL.format(n=n, last=n - 1, indices=str(indices))

        # Build multimodal content: images + text prompt
        parts = [
            genai_types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")
            for _, jpeg in self._frame_buffer
        ]
        parts.append(genai_types.Part(text=prompt_text))

        try:
            from agent.token_logger import log_call, extract_token_counts
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=genai_types.Content(parts=parts, role="user"),
                config={
                    "system_instruction": _SYSTEM,
                    "temperature": 0.1,
                    "max_output_tokens": 16384,
                },
            )
            inp, out = extract_token_counts(response)
            log_call(
                self.model_name, "GeminiVisionAnalyzer", inp, out,
                extra={"job_id": self.job_id, "frames_analysed": n},
            )
            print(
                f"[GeminiVisionAnalyzer] {n} frames → "
                f"{inp:,} in / {out:,} out tokens",
                flush=True,
            )

            text = response.text.strip()
            self._results = _parse_json_robust(text, indices)

        except Exception as exc:
            logger.error("[GeminiVisionAnalyzer] Call failed: %s", exc)
            print(f"[GeminiVisionAnalyzer] Fallback due to error: {exc}", flush=True)
            self._results = [_fallback_frame(idx) for idx in indices]

    # ── Facial Expression Summary ─────────────────────────────────────────────

    def get_facial_summary(self) -> dict:
        """Return facial analysis summary matching FacialExpressionAnalyzer.get_summary()."""
        self._run_analysis()
        frames = self._results
        total = len(frames)
        faces_detected = sum(1 for f in frames if f.get("face_detected"))
        face_count = faces_detected or 1  # avoid div-by-zero

        # Dominant emotion distribution
        emo_counts: dict[str, int] = {}
        for f in frames:
            if f.get("face_detected"):
                emo = f.get("dominant_emotion", "neutral")
                emo_counts[emo] = emo_counts.get(emo, 0) + 1
        emo_dist = {k: round(v / face_count, 3) for k, v in emo_counts.items()}
        if not emo_dist:
            emo_dist = {"neutral": 1.0}

        # Average probabilities per emotion
        emo_keys = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]
        avg_emos = {
            k: round(sum(f.get("emotions", {}).get(k, 0.0) for f in frames) / (total or 1), 3)
            for k in emo_keys
        }

        # Smile ratio
        smile_count = sum(
            1 for f in frames if f.get("smile_detected") and f.get("face_detected")
        )
        smile_ratio = round(smile_count / face_count, 3)

        # Scenario scores via impact weights
        scenario_scores = {}
        for scenario, impacts in _SCENARIO_IMPACT.items():
            score = sum(emo_dist.get(e, 0) * impacts.get(e, 0.3) for e in emo_dist)
            scenario_scores[scenario] = round(min(score * 100, 100), 2)

        # Timeline and off-segments
        timeline = [
            (f["frame_idx"], f.get("dominant_emotion", "neutral"))
            for f in frames if f.get("face_detected")
        ]
        bad_emos = {"sad", "angry", "fear", "disgust"}
        off_segments = [
            {
                "frame_idx": f["frame_idx"],
                "dominant_emotion": f.get("dominant_emotion"),
                "reason": "negative facial emotion",
            }
            for f in frames
            if f.get("face_detected") and f.get("dominant_emotion") in bad_emos
        ]

        return {
            "total_frames_analyzed": total,
            "faces_detected": faces_detected,
            "emotion_distribution": emo_dist,
            "average_emotion_scores": avg_emos,
            "smile_ratio": smile_ratio,
            "scenario_scores": scenario_scores,
            "emotion_timeline": timeline[:40],
            "off_segments": off_segments,
        }

    # ── Body Language Summary ─────────────────────────────────────────────────

    def get_body_summary(self) -> dict:
        """Return body language summary matching BodyLanguageAnalyzer.get_summary()."""
        self._run_analysis()
        frames = self._results
        total = len(frames)
        pose_detected_count = sum(1 for f in frames if f.get("pose_detected"))

        avg_metrics: dict = {}
        off_frames: list = []

        if pose_detected_count > 0:
            detected = [f for f in frames if f.get("pose_detected")]
            postures  = [f.get("posture", {}) for f in detected]
            conf_sigs = [f.get("confidence_signals", {}) for f in detected]

            def _avg(dicts, key, default=0.5):
                return round(sum(d.get(key, default) for d in dicts) / len(dicts), 3)

            avg_metrics = {
                "avg_shoulder_alignment": _avg(postures, "shoulder_alignment"),
                "avg_head_uprightness":   _avg(postures, "head_uprightness"),
                "avg_openness":           _avg(postures, "openness"),
                "avg_confidence_score":   _avg(conf_sigs, "confidence_score"),
                "arms_crossed_ratio": round(
                    sum(1 for c in conf_sigs if c.get("arms_crossed")) / len(conf_sigs), 3
                ),
            }

            for f in detected:
                conf = f.get("confidence_signals", {}).get("confidence_score", 0.5)
                if conf < 0.45:
                    off_frames.append({
                        "frame_idx": f["frame_idx"],
                        "reason": "low confidence posture",
                        "confidence_score": round(conf, 3),
                    })
                elif f.get("confidence_signals", {}).get("arms_crossed"):
                    off_frames.append({
                        "frame_idx": f["frame_idx"],
                        "reason": "arms crossed",
                    })

        scenario_scores = _body_scenario_scores(avg_metrics)

        return {
            "total_frames_analyzed": total,
            "pose_detected_count": pose_detected_count,
            "average_metrics": avg_metrics,
            "scenario_scores": scenario_scores,
            "off_frames": off_frames,
        }

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def release(self):
        """No native resources to release — clears internal buffers."""
        self._frame_buffer.clear()
        self._results.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json_robust(text: str, indices: list[int]) -> list[dict]:
    """
    Parse the Gemini response JSON array as robustly as possible.
    1. Strip markdown fences.
    2. Try full parse of the outermost array.
    3. If that fails (e.g. truncated response), extract every complete
       JSON object individually and fill missing frame slots with fallbacks.
    """
    # Strip markdown fences
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text.strip())

    # Try to extract outermost array
    m = re.search(r"\[.*\]", text, re.DOTALL)
    candidate = m.group(0) if m else text

    # Attempt full parse first
    try:
        results = json.loads(candidate)
        if isinstance(results, list) and results:
            return results
    except json.JSONDecodeError:
        pass

    # Partial recovery: extract every complete top-level {...} object
    recovered: list[dict] = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start : i + 1])
                    recovered.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None

    if recovered:
        recovered_by_idx = {obj.get("frame_idx"): obj for obj in recovered}
        results = [recovered_by_idx.get(idx, _fallback_frame(idx)) for idx in indices]
        n_recovered = len(recovered)
        n_fallback = len(indices) - n_recovered
        print(
            f"[GeminiVisionAnalyzer] Partial parse recovered {n_recovered}/{len(indices)} frames"
            + (f" — {n_fallback} filled with fallbacks" if n_fallback else ""),
            flush=True,
        )
        return results

    # Total failure
    print(f"[GeminiVisionAnalyzer] JSON parse failed entirely — using all fallbacks", flush=True)
    return [_fallback_frame(idx) for idx in indices]


def _fallback_frame(frame_idx: int) -> dict:
    return {
        "frame_idx": frame_idx,
        "face_detected": False,
        "dominant_emotion": "neutral",
        "emotions": {
            "happy": 0.0, "sad": 0.0, "angry": 0.0, "fear": 0.0,
            "surprise": 0.0, "disgust": 0.0, "neutral": 1.0,
        },
        "smile_detected": False,
        "pose_detected": False,
        "posture": {
            "shoulder_alignment": 0.5, "head_uprightness": 0.5,
            "spine_straightness": 0.5, "openness": 0.5,
        },
        "confidence_signals": {
            "arms_crossed": False, "lean_direction": "centered",
            "forward_lean": 0.0, "confidence_score": 0.5,
        },
    }


def _body_scenario_scores(am: dict) -> dict:
    if not am:
        return {"job_interview": 50.0, "business_deal": 50.0, "date": 50.0}
    posture  = (am.get("avg_shoulder_alignment", 0.5) + am.get("avg_head_uprightness", 0.5)) / 2
    conf     = am.get("avg_confidence_score", 0.5)
    openness = am.get("avg_openness", 0.5)
    arms_ok  = 1.0 - am.get("arms_crossed_ratio", 0.0)
    return {
        "job_interview": round(min((posture * 0.30 + conf * 0.35 + openness * 0.20 + arms_ok * 0.15) * 100, 100), 2),
        "business_deal": round(min((posture * 0.25 + conf * 0.40 + openness * 0.20 + arms_ok * 0.15) * 100, 100), 2),
        "date":          round(min((posture * 0.20 + conf * 0.20 + openness * 0.35 + arms_ok * 0.25) * 100, 100), 2),
    }
