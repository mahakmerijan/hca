"""
Voice & Speech Analyzer
Analyzes audio from video for tone, pace, energy, clarity, and speech content.
"""

import os
import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False


class VoiceSpeechAnalyzer:
    """Analyzes voice characteristics and speech patterns from audio."""

    def __init__(self, segment_duration: float = 5.0):
        """
        Args:
            segment_duration: Duration in seconds for each analysis segment.
        """
        self.segment_duration = segment_duration
        self.audio_features = {}
        self.transcript = ""
        self.speech_metrics = {}

    def load_audio(self, audio_path: str) -> bool:
        """Load audio file for analysis."""
        if not os.path.exists(audio_path):
            print(f"[VoiceSpeechAnalyzer] Audio file not found: {audio_path}")
            return False

        if not LIBROSA_AVAILABLE:
            print("[VoiceSpeechAnalyzer] librosa not available. Skipping audio analysis.")
            return False

        try:
            self.y, self.sr_rate = librosa.load(audio_path, sr=22050)
            self.duration = librosa.get_duration(y=self.y, sr=self.sr_rate)
            return True
        except Exception as e:
            print(f"[VoiceSpeechAnalyzer] Failed to load audio: {e}")
            return False

    def analyze_pitch(self) -> dict:
        """Analyze pitch characteristics (fundamental frequency)."""
        if not LIBROSA_AVAILABLE or not hasattr(self, "y"):
            return {"mean_pitch": 0, "pitch_variation": 0, "pitch_range": 0}

        try:
            # Extract pitch using librosa
            pitches, magnitudes = librosa.piptrack(y=self.y, sr=self.sr_rate)
            # Get the most prominent pitch at each time step
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if not pitch_values:
                return {"mean_pitch": 0, "pitch_variation": 0, "pitch_range": 0}

            pitch_array = np.array(pitch_values)
            return {
                "mean_pitch": round(float(np.mean(pitch_array)), 2),
                "pitch_variation": round(float(np.std(pitch_array)), 2),
                "pitch_range": round(float(np.max(pitch_array) - np.min(pitch_array)), 2),
                "pitch_stability": round(
                    max(0, 1.0 - float(np.std(pitch_array)) / (float(np.mean(pitch_array)) + 1e-8)),
                    3,
                ),
            }
        except Exception as e:
            print(f"[VoiceSpeechAnalyzer] Pitch analysis failed: {e}")
            return {"mean_pitch": 0, "pitch_variation": 0, "pitch_range": 0}

    def analyze_energy(self) -> dict:
        """Analyze vocal energy and volume patterns."""
        if not LIBROSA_AVAILABLE or not hasattr(self, "y"):
            return {"mean_energy": 0, "energy_variation": 0}

        try:
            rms = librosa.feature.rms(y=self.y)[0]
            return {
                "mean_energy": round(float(np.mean(rms)), 5),
                "energy_variation": round(float(np.std(rms)), 5),
                "max_energy": round(float(np.max(rms)), 5),
                "energy_consistency": round(
                    max(0, 1.0 - float(np.std(rms)) / (float(np.mean(rms)) + 1e-8)),
                    3,
                ),
                "dynamic_range": round(float(np.max(rms) - np.min(rms)), 5),
            }
        except Exception as e:
            print(f"[VoiceSpeechAnalyzer] Energy analysis failed: {e}")
            return {"mean_energy": 0, "energy_variation": 0}

    def analyze_speaking_rate(self) -> dict:
        """Estimate speaking rate from audio onset detection."""
        if not LIBROSA_AVAILABLE or not hasattr(self, "y"):
            return {"estimated_syllables_per_sec": 0, "pace_score": 0.5}

        try:
            onset_frames = librosa.onset.onset_detect(y=self.y, sr=self.sr_rate)
            onset_times = librosa.frames_to_time(onset_frames, sr=self.sr_rate)
            num_onsets = len(onset_times)
            syllables_per_sec = num_onsets / self.duration if self.duration > 0 else 0

            # Ideal speaking rate is ~3-5 syllables/sec
            if 3.0 <= syllables_per_sec <= 5.0:
                pace_score = 0.9
            elif 2.0 <= syllables_per_sec < 3.0 or 5.0 < syllables_per_sec <= 6.5:
                pace_score = 0.7
            elif 1.0 <= syllables_per_sec < 2.0 or 6.5 < syllables_per_sec <= 8.0:
                pace_score = 0.5
            else:
                pace_score = 0.3

            return {
                "estimated_syllables_per_sec": round(syllables_per_sec, 2),
                "num_onsets": num_onsets,
                "duration_seconds": round(self.duration, 2),
                "pace_score": pace_score,
                "pace_label": self._pace_label(syllables_per_sec),
            }
        except Exception as e:
            print(f"[VoiceSpeechAnalyzer] Speaking rate analysis failed: {e}")
            return {"estimated_syllables_per_sec": 0, "pace_score": 0.5}

    def _pace_label(self, sps: float) -> str:
        if sps < 2.0:
            return "very_slow"
        elif sps < 3.0:
            return "slow"
        elif sps <= 5.0:
            return "ideal"
        elif sps <= 6.5:
            return "fast"
        else:
            return "very_fast"

    def analyze_pauses(self) -> dict:
        """Detect and analyze pauses in speech."""
        if not LIBROSA_AVAILABLE or not hasattr(self, "y"):
            return {"num_pauses": 0, "avg_pause_duration": 0, "pause_ratio": 0}

        try:
            # Detect silent intervals
            intervals = librosa.effects.split(self.y, top_db=30)
            if len(intervals) < 2:
                return {"num_pauses": 0, "avg_pause_duration": 0, "pause_ratio": 0}

            pauses = []
            for i in range(1, len(intervals)):
                gap_start = intervals[i - 1][1]
                gap_end = intervals[i][0]
                pause_dur = (gap_end - gap_start) / self.sr_rate
                if pause_dur > 0.2:  # Only count pauses > 200ms
                    pauses.append(pause_dur)

            total_pause = sum(pauses)
            pause_ratio = total_pause / self.duration if self.duration > 0 else 0

            # Moderate pauses are good (thinking, emphasis); too many or too long are bad
            if 0.1 <= pause_ratio <= 0.25:
                pause_score = 0.85
            elif 0.05 <= pause_ratio < 0.1 or 0.25 < pause_ratio <= 0.4:
                pause_score = 0.65
            else:
                pause_score = 0.4

            return {
                "num_pauses": len(pauses),
                "avg_pause_duration": round(np.mean(pauses) if pauses else 0, 3),
                "total_pause_time": round(total_pause, 3),
                "pause_ratio": round(pause_ratio, 3),
                "pause_score": pause_score,
            }
        except Exception as e:
            print(f"[VoiceSpeechAnalyzer] Pause analysis failed: {e}")
            return {"num_pauses": 0, "avg_pause_duration": 0, "pause_ratio": 0}

    def transcribe_speech(self, audio_path: str) -> str:
        """Transcribe speech to text using SpeechRecognition."""
        if not SR_AVAILABLE:
            print("[VoiceSpeechAnalyzer] SpeechRecognition not available.")
            return ""

        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            self.transcript = text
            return text
        except sr.UnknownValueError:
            print("[VoiceSpeechAnalyzer] Could not understand audio.")
            return ""
        except sr.RequestError as e:
            print(f"[VoiceSpeechAnalyzer] Transcription service error: {e}")
            return ""
        except Exception as e:
            print(f"[VoiceSpeechAnalyzer] Transcription failed: {e}")
            return ""

    def analyze_speech_content(self) -> dict:
        """Analyze transcribed speech for filler words, vocabulary, etc."""
        if not self.transcript:
            return {
                "word_count": 0,
                "filler_word_ratio": 0,
                "vocabulary_richness": 0,
                "content_score": 0.5,
            }

        words = self.transcript.lower().split()
        word_count = len(words)

        # Count filler words
        fillers = {"um", "uh", "like", "you know", "basically", "actually",
                   "literally", "honestly", "right", "so", "well", "er", "ah"}
        filler_count = sum(1 for w in words if w in fillers)
        filler_ratio = filler_count / word_count if word_count > 0 else 0

        # Vocabulary richness (unique words / total words)
        unique_words = len(set(words))
        vocab_richness = unique_words / word_count if word_count > 0 else 0

        # Positive/confident language
        confident_words = {"confident", "certainly", "absolutely", "definitely",
                           "achieve", "successful", "excellent", "great",
                           "experience", "skills", "passion", "dedicated",
                           "innovative", "creative", "leadership", "team"}
        confident_count = sum(1 for w in words if w in confident_words)
        confidence_word_ratio = confident_count / word_count if word_count > 0 else 0

        # Content score
        content_score = 0.5
        content_score += (1.0 - min(filler_ratio * 5, 0.5)) * 0.3  # Fewer fillers = better
        content_score += min(vocab_richness, 1.0) * 0.2
        content_score += min(confidence_word_ratio * 10, 0.3)

        return {
            "word_count": word_count,
            "filler_word_count": filler_count,
            "filler_word_ratio": round(filler_ratio, 3),
            "vocabulary_richness": round(vocab_richness, 3),
            "confident_language_ratio": round(confidence_word_ratio, 3),
            "content_score": round(min(content_score, 1.0), 3),
            "transcript_preview": self.transcript[:200] + "..." if len(self.transcript) > 200 else self.transcript,
        }

    def run_full_analysis(self, audio_path: str) -> dict:
        """Run all voice and speech analyses."""
        loaded = self.load_audio(audio_path)

        pitch = self.analyze_pitch() if loaded else {}
        energy = self.analyze_energy() if loaded else {}
        speaking_rate = self.analyze_speaking_rate() if loaded else {}
        pauses = self.analyze_pauses() if loaded else {}

        # Transcription
        self.transcribe_speech(audio_path)
        speech_content = self.analyze_speech_content()

        self.audio_features = {
            "pitch": pitch,
            "energy": energy,
            "speaking_rate": speaking_rate,
            "pauses": pauses,
            "speech_content": speech_content,
            "audio_loaded": loaded,
        }

        return self.audio_features

    def get_voice_score(self) -> dict:
        """
        Compute overall voice/speech scores for each scenario.

        Returns:
            Dict with scores for job_interview, business_deal, date (0-100).
        """
        if not self.audio_features or not self.audio_features.get("audio_loaded"):
            return {"job_interview": 50.0, "business_deal": 50.0, "date": 50.0}

        pitch = self.audio_features.get("pitch", {})
        energy = self.audio_features.get("energy", {})
        rate = self.audio_features.get("speaking_rate", {})
        pauses = self.audio_features.get("pauses", {})
        content = self.audio_features.get("speech_content", {})

        pitch_stability = pitch.get("pitch_stability", 0.5)
        energy_consistency = energy.get("energy_consistency", 0.5)
        pace_score = rate.get("pace_score", 0.5)
        pause_score = pauses.get("pause_score", 0.5)
        content_score = content.get("content_score", 0.5)

        # Job interview: clarity and content matter most
        job_score = (pitch_stability * 0.15 + energy_consistency * 0.15 +
                     pace_score * 0.25 + pause_score * 0.15 + content_score * 0.30) * 100

        # Business deal: energy and confidence matter most
        biz_score = (pitch_stability * 0.15 + energy_consistency * 0.20 +
                     pace_score * 0.20 + pause_score * 0.15 + content_score * 0.30) * 100

        # Date: warmth and natural flow matter most
        date_score = (pitch_stability * 0.20 + energy_consistency * 0.20 +
                      pace_score * 0.25 + pause_score * 0.20 + content_score * 0.15) * 100

        return {
            "job_interview": round(min(job_score, 100), 2),
            "business_deal": round(min(biz_score, 100), 2),
            "date": round(min(date_score, 100), 2),
        }

    def get_summary(self) -> dict:
        """Get full summary of voice/speech analysis."""
        return {
            "audio_features": self.audio_features,
            "scenario_scores": self.get_voice_score(),
            "transcript_available": bool(self.transcript),
            "full_transcript": self.transcript,
        }
