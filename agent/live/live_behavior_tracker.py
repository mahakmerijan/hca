"""
LiveBehaviorTracker
=====================
Wraps the existing per-frame analyzers (facial expression, body language,
eyebrow/face-landmark blendshapes) plus per-chunk voice analysis so they
can be fed live webcam frames / mic chunks during a Live Digital Twin
conversation, instead of a single pre-recorded file.

All three visual analyzers already accumulate state internally across
`analyze_frame()` calls, which is exactly the shape we need for a live
stream — we just call them once per captured frame.
"""

import base64
import os
import tempfile
from typing import Optional

import numpy as np


class LiveBehaviorTracker:
    def __init__(self):
        self._facial = None
        self._body = None
        self._face_landmarks = None
        self.voice_chunks: list[dict] = []
        self.transcript_fragments: list[str] = []
        self._frame_idx = 0
        # Frame-index boundaries tagged to each conversation turn, so the judge
        # can be given "what was your face/posture doing right when you said X".
        self.turn_marks: list[dict] = []

    # ── lazy analyzer init (heavy imports / model downloads) ──────────
    def _ensure_visual_analyzers(self):
        if self._facial is None:
            from agent.analyzers.facial_expression import FacialExpressionAnalyzer
            self._facial = FacialExpressionAnalyzer(0.5)
        if self._body is None:
            from agent.analyzers.body_language import BodyLanguageAnalyzer
            self._body = BodyLanguageAnalyzer(0.5)
        if self._face_landmarks is None:
            from agent.analyzers.face_landmark_analyzer import FaceLandmarkAnalyzer
            self._face_landmarks = FaceLandmarkAnalyzer(0.5)

    # ── ingestion ───────────────────────────────────────────────────
    def mark_turn(self, role: str, content: str):
        """Record the current frame boundary as the start of a new conversation turn."""
        self.turn_marks.append({"frame_idx": self._frame_idx, "role": role, "content": content})

    def ingest_frame_b64(self, image_b64: str) -> bool:
        """Decode a base64 JPEG/PNG frame (e.g. from a <canvas>.toDataURL()) and analyze it."""
        try:
            import cv2
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]
            raw = base64.b64decode(image_b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return False
        except Exception as e:
            print(f"[LiveBehaviorTracker] Frame decode failed: {e}")
            return False

        self._ensure_visual_analyzers()
        idx = self._frame_idx
        self._frame_idx += 1
        try:
            self._facial.analyze_frame(frame, idx)
            self._body.analyze_frame(frame, idx)
            self._face_landmarks.analyze_frame(frame, idx)
            return True
        except Exception as e:
            print(f"[LiveBehaviorTracker] Frame analysis failed: {e}")
            return False

    def ingest_audio_b64(self, audio_b64: str, suffix: str = ".webm") -> Optional[dict]:
        """Decode a base64 audio chunk, run voice analysis on it, and accumulate the result."""
        try:
            if "," in audio_b64:
                audio_b64 = audio_b64.split(",", 1)[1]
            raw = base64.b64decode(audio_b64)
        except Exception as e:
            print(f"[LiveBehaviorTracker] Audio decode failed: {e}")
            return None

        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_in.write(raw)
        tmp_in.close()
        wav_path = tmp_in.name
        converted = None
        try:
            # librosa/SpeechRecognition need a decodable waveform — convert via pydub if not already wav
            if suffix != ".wav":
                from pydub import AudioSegment
                converted = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                converted.close()
                AudioSegment.from_file(wav_path).export(converted.name, format="wav")
                wav_path = converted.name

            from agent.analyzers.voice_speech import VoiceSpeechAnalyzer
            analyzer = VoiceSpeechAnalyzer(segment_duration=5.0)
            result = analyzer.run_full_analysis(wav_path)
            if analyzer.transcript:
                self.transcript_fragments.append(analyzer.transcript)
            self.voice_chunks.append(result)
            return result
        except Exception as e:
            print(f"[LiveBehaviorTracker] Audio chunk analysis failed: {e}")
            return None
        finally:
            for p in (tmp_in.name, converted.name if converted else None):
                if p:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    # ── aggregation for the judge / live behavior_hint ─────────────
    def get_aggregated_summary(self) -> dict:
        facial_summary = self._facial.get_summary() if self._facial else {}
        body_summary = self._body.get_summary() if self._body else {}
        brow_summary = self._face_landmarks.get_summary() if self._face_landmarks else {}

        voice_summary = self._aggregate_voice()

        return {
            "facial_expression": facial_summary,
            "body_language": body_summary,
            "eyebrow_and_face": brow_summary,
            "voice": voice_summary,
            "full_transcript": " ".join(self.transcript_fragments).strip(),
        }

    def get_turn_segments(self) -> list:
        """
        Break the per-frame facial/body/eyebrow history into segments bounded by
        each conversation turn, so the judge can ground "at this point you..."
        feedback in the exact quote it happened around, instead of only an
        aggregate for the whole call.
        """
        if not self.turn_marks:
            return []

        facial_frames = self._facial.frame_emotions if self._facial else []
        body_frames = self._body.frame_results if self._body else []
        brow_frames = self._face_landmarks.frame_results if self._face_landmarks else []

        segments = []
        marks = self.turn_marks
        for i, mark in enumerate(marks):
            if mark["role"] != "user":
                continue
            start = mark["frame_idx"]
            end = marks[i + 1]["frame_idx"] if i + 1 < len(marks) else self._frame_idx

            f_slice = [f for f in facial_frames if start <= f.get("frame_idx", -1) < end and f.get("face_detected")]
            b_slice = [b for b in body_frames if start <= b.get("frame_idx", -1) < end and b.get("pose_detected")]
            e_slice = [e for e in brow_frames if start <= e.get("frame_idx", -1) < end and e.get("face_detected")]

            dominant_emotions = [f["dominant_emotion"] for f in f_slice if f.get("dominant_emotion")]
            avg_confidence = (
                round(float(np.mean([b["confidence_signals"].get("confidence_score", 0.5) for b in b_slice])), 3)
                if b_slice else None
            )
            avg_openness = (
                round(float(np.mean([b["posture"].get("openness", 0.5) for b in b_slice])), 3)
                if b_slice else None
            )
            arms_crossed_ratio = (
                round(sum(1 for b in b_slice if b["confidence_signals"].get("arms_crossed")) / len(b_slice), 3)
                if b_slice else None
            )
            avg_brow_raise = (
                round(float(np.mean([
                    (e["blendshapes"].get("browOuterUpLeft", 0) + e["blendshapes"].get("browOuterUpRight", 0)) / 2
                    for e in e_slice
                ])), 4) if e_slice else None
            )
            avg_brow_furrow = (
                round(float(np.mean([
                    (e["blendshapes"].get("browDownLeft", 0) + e["blendshapes"].get("browDownRight", 0)) / 2
                    for e in e_slice
                ])), 4) if e_slice else None
            )

            segments.append({
                "turn_index": i,
                "user_said": mark["content"],
                "frames_captured": len(f_slice) + len(b_slice) + len(e_slice),
                "dominant_emotion": max(set(dominant_emotions), key=dominant_emotions.count) if dominant_emotions else None,
                "avg_confidence_score": avg_confidence,
                "avg_openness": avg_openness,
                "arms_crossed_ratio": arms_crossed_ratio,
                "avg_eyebrow_raise": avg_brow_raise,
                "avg_eyebrow_furrow": avg_brow_furrow,
            })
        return segments

    def _aggregate_voice(self) -> dict:
        if not self.voice_chunks:
            return {}
        pitches = [c.get("pitch", {}).get("mean_pitch", 0) for c in self.voice_chunks if c.get("pitch")]
        energies = [c.get("energy", {}).get("mean_energy", 0) for c in self.voice_chunks if c.get("energy")]
        rates = [c.get("speaking_rate", {}).get("estimated_syllables_per_sec", 0) for c in self.voice_chunks if c.get("speaking_rate")]
        pace_labels = [c.get("speaking_rate", {}).get("pace_label") for c in self.voice_chunks if c.get("speaking_rate", {}).get("pace_label")]
        pause_ratios = [c.get("pauses", {}).get("pause_ratio", 0) for c in self.voice_chunks if c.get("pauses")]
        filler_ratios = [c.get("speech_content", {}).get("filler_word_ratio", 0) for c in self.voice_chunks if c.get("speech_content")]

        def _avg(lst):
            return round(float(np.mean(lst)), 3) if lst else 0

        return {
            "chunks_analyzed": len(self.voice_chunks),
            "avg_pitch": _avg(pitches),
            "avg_energy": _avg(energies),
            "avg_syllables_per_sec": _avg(rates),
            "dominant_pace": max(set(pace_labels), key=pace_labels.count) if pace_labels else "unknown",
            "avg_pause_ratio": _avg(pause_ratios),
            "avg_filler_word_ratio": _avg(filler_ratios),
        }

    def get_behavior_hint(self) -> str:
        """Short natural-language snapshot for the twin chat agent to subtly react to."""
        agg = self.get_aggregated_summary()
        bits = []
        conf = agg["body_language"].get("average_metrics", {}).get("avg_confidence_score")
        if conf is not None:
            bits.append("confident posture" if conf >= 0.6 else "tense/closed posture" if conf < 0.4 else "neutral posture")
        smile = agg["facial_expression"].get("smile_ratio")
        if smile is not None:
            bits.append("smiling often" if smile > 0.3 else "mostly neutral/serious expression")
        pace = agg["voice"].get("dominant_pace")
        if pace and pace != "unknown":
            bits.append(f"speaking pace: {pace}")
        return "; ".join(bits)

    def release(self):
        if self._body:
            try:
                self._body.release()
            except Exception:
                pass
        if self._face_landmarks:
            try:
                self._face_landmarks.release()
            except Exception:
                pass
