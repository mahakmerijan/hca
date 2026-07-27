"""
Voice & Speech Analyzer (Gemini Edition)
==========================================
Replaces librosa / SpeechRecognition with a single Gemini audio API call.
Sends the WAV file as inline audio data → Gemini returns transcription,
pitch, pace, energy, filler-word ratio, vocabulary richness, etc.

Memory: <1 MB (just the audio bytes) vs ~50 MB for librosa.
Token usage is logged to logs/token_usage.jsonl.
"""

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    genai = None
    genai_types = None
    _GENAI_AVAILABLE = False

_AUDIO_PROMPT = """\
Analyse this audio recording thoroughly. Return ONLY valid JSON — no markdown fences, no explanation.

{
  "transcript": "<full verbatim transcription — empty string if no speech detected>",
  "word_count": <integer>,
  "speaking_rate": {
    "pace_label": "<very_slow|slow|ideal|fast|very_fast>",
    "estimated_words_per_min": <integer>,
    "estimated_syllables_per_sec": <float>
  },
  "pitch": {
    "mean_pitch": <Hz as integer, typical male 80-180 Hz, female 160-300 Hz>,
    "pitch_variation": <0.0-1.0>,
    "pitch_stability": <0.0-1.0, 1=very consistent>
  },
  "energy": {
    "mean_energy": <0.0-1.0>,
    "energy_variation": <0.0-1.0>,
    "energy_label": "<low|moderate|high>"
  },
  "pauses": {
    "num_pauses": <integer count of noticeable pauses>,
    "avg_pause_duration": <seconds as float>,
    "pause_ratio": <0.0-1.0, fraction of total time spent in silence>
  },
  "filler_word_ratio": <0.0-1.0, e.g. 0.03 = 3 per 100 words>,
  "vocabulary_richness": <0.0-1.0, unique/total word ratio>,
  "confident_language_ratio": <0.0-1.0>,
  "tone_label": "<professional|casual|nervous|assertive|warm>",
  "scenario_scores": {
    "job_interview": <0-100>,
    "business_deal": <0-100>,
    "date": <0-100>
  }
}
"""



class VoiceSpeechAnalyzer:
    """
    Voice & speech analyser powered by Gemini audio API.
    Sends the extracted WAV file to Gemini and receives transcription,
    pitch, pace, energy, filler-word ratio, vocabulary, and scenario scores.
    """

    def __init__(self, segment_duration: float = 5.0, job_id: str = ""):
        self.segment_duration = segment_duration  # kept for API compatibility
        self.job_id = job_id
        self.model_name = os.getenv("LLM_MODEL", "gemini-2.5-flash")
        self.audio_features: dict = {}
        self.transcript: str = ""
        self._client = None

        if _GENAI_AVAILABLE:
            try:
                self._client = genai.Client(
                    vertexai=True,
                    project=os.getenv("VERTEX_PROJECT", "ai-ml-integrations"),
                    location=os.getenv("VERTEX_LOCATION", "us-central1"),
                )
            except Exception as exc:
                logger.warning("[VoiceSpeechAnalyzer] Client init failed: %s", exc)

    # ── Core analysis ─────────────────────────────────────────────────────────

    def run_full_analysis(self, audio_path: str) -> dict:
        """Send audio to Gemini for full voice & speech analysis."""
        if not os.path.exists(audio_path):
            print(f"[VoiceSpeechAnalyzer] Audio file not found: {audio_path}", flush=True)
            self.audio_features = _fallback_audio_features()
            return self.audio_features

        if self._client is None or not _GENAI_AVAILABLE:
            print("[VoiceSpeechAnalyzer] No Gemini client — using fallback.", flush=True)
            self.audio_features = _fallback_audio_features()
            return self.audio_features

        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            from agent.token_logger import log_call, extract_token_counts
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=genai_types.Content(
                    parts=[
                        genai_types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                        genai_types.Part(text=_AUDIO_PROMPT),
                    ],
                    role="user",
                ),
                config={"temperature": 0.1},
            )
            inp, out = extract_token_counts(response)
            log_call(
                self.model_name, "GeminiVoiceAnalyzer", inp, out,
                extra={"job_id": self.job_id, "audio_bytes": len(audio_bytes)},
            )
            print(
                f"[VoiceSpeechAnalyzer] Audio analysed — "
                f"{inp:,} in / {out:,} out tokens",
                flush=True,
            )

            text = response.text.strip()
            text = re.sub(r"^```[a-z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text.strip())
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                text = m.group(0)
            data = json.loads(text)

        except Exception as exc:
            logger.error("[VoiceSpeechAnalyzer] API call failed: %s", exc)
            print(f"[VoiceSpeechAnalyzer] Fallback due to error: {exc}", flush=True)
            self.audio_features = _fallback_audio_features()
            return self.audio_features

        # Map Gemini response → existing audio_features structure
        sr_data = data.get("speaking_rate", {})
        self.transcript = data.get("transcript", "")
        word_count = data.get("word_count", len(self.transcript.split()) if self.transcript else 0)
        filler_ratio = data.get("filler_word_ratio", 0.0)
        vocab_richness = data.get("vocabulary_richness", 0.5)
        conf_lang = data.get("confident_language_ratio", 0.5)

        self.audio_features = {
            "audio_loaded": True,
            "pitch": {
                "mean_pitch":       data.get("pitch", {}).get("mean_pitch", 150),
                "pitch_variation":  data.get("pitch", {}).get("pitch_variation", 0.3),
                "pitch_stability":  data.get("pitch", {}).get("pitch_stability", 0.7),
            },
            "energy": data.get("energy", {"mean_energy": 0.5, "energy_variation": 0.3, "energy_label": "moderate"}),
            "speaking_rate": {
                "pace_label":               sr_data.get("pace_label", "ideal"),
                "estimated_syllables_per_sec": sr_data.get("estimated_syllables_per_sec", 3.5),
                "estimated_words_per_min":  sr_data.get("estimated_words_per_min", 130),
                "pace_score": _pace_score(sr_data.get("pace_label", "ideal")),
            },
            "pauses": {
                "num_pauses":        data.get("pauses", {}).get("num_pauses", 0),
                "avg_pause_duration": data.get("pauses", {}).get("avg_pause_duration", 0.0),
                "pause_ratio":       data.get("pauses", {}).get("pause_ratio", 0.1),
                "pause_score":       0.75,
            },
            "speech_content": {
                "word_count":               word_count,
                "filler_word_ratio":        round(filler_ratio, 3),
                "vocabulary_richness":      round(vocab_richness, 3),
                "confident_language_ratio": round(conf_lang, 3),
                "content_score":            round(
                    0.5 + (1.0 - min(filler_ratio * 5, 0.5)) * 0.3
                        + min(vocab_richness, 1.0) * 0.2
                        + min(conf_lang * 10, 0.3),
                    3,
                ),
                "transcript_preview": self.transcript[:200] + ("…" if len(self.transcript) > 200 else ""),
            },
        }
        return self.audio_features

    # ── Summary (same format as old analyzer) ────────────────────────────────

    def get_summary(self) -> dict:
        return {
            "audio_features":       self.audio_features,
            "scenario_scores":      self._scenario_scores(),
            "transcript_available": bool(self.transcript),
            "full_transcript":      self.transcript,
            "dialogue_issues":      [],   # Gemini counsellor handles dialogue coaching
        }

    def _scenario_scores(self) -> dict:
        if not self.audio_features or not self.audio_features.get("audio_loaded"):
            return {"job_interview": 50.0, "business_deal": 50.0, "date": 50.0}
        pitch_stab  = self.audio_features.get("pitch", {}).get("pitch_stability", 0.5)
        energy_cons = 1.0 - self.audio_features.get("energy", {}).get("energy_variation", 0.3)
        pace_sc     = self.audio_features.get("speaking_rate", {}).get("pace_score", 0.5)
        pause_sc    = self.audio_features.get("pauses", {}).get("pause_score", 0.5)
        content_sc  = self.audio_features.get("speech_content", {}).get("content_score", 0.5)
        job = (pitch_stab * 0.15 + energy_cons * 0.15 + pace_sc * 0.25 + pause_sc * 0.15 + content_sc * 0.30) * 100
        biz = (pitch_stab * 0.15 + energy_cons * 0.20 + pace_sc * 0.20 + pause_sc * 0.15 + content_sc * 0.30) * 100
        date = (pitch_stab * 0.20 + energy_cons * 0.20 + pace_sc * 0.25 + pause_sc * 0.20 + content_sc * 0.15) * 100
        return {
            "job_interview": round(min(job,  100), 2),
            "business_deal": round(min(biz,  100), 2),
            "date":          round(min(date, 100), 2),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fallback_audio_features() -> dict:
    return {
        "audio_loaded": False,
        "pitch": {"mean_pitch": 0, "pitch_variation": 0, "pitch_stability": 0.5},
        "energy": {"mean_energy": 0, "energy_variation": 0, "energy_label": "moderate"},
        "speaking_rate": {"pace_label": "ideal", "estimated_syllables_per_sec": 0,
                          "estimated_words_per_min": 0, "pace_score": 0.5},
        "pauses": {"num_pauses": 0, "avg_pause_duration": 0, "pause_ratio": 0, "pause_score": 0.5},
        "speech_content": {"word_count": 0, "filler_word_ratio": 0,
                            "vocabulary_richness": 0.5, "confident_language_ratio": 0.5,
                            "content_score": 0.5, "transcript_preview": ""},
    }


def _pace_score(label: str) -> float:
    return {"very_slow": 0.3, "slow": 0.6, "ideal": 0.9, "fast": 0.65, "very_fast": 0.35}.get(label, 0.5)
