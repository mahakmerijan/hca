"""
Facial Expression Analyzer
Detects emotions from facial expressions in video frames using DeepFace.
"""

import numpy as np
from deepface import DeepFace


# Mapping emotions to positivity scores (0-1 scale)
EMOTION_SCORES = {
    "happy": 0.95,
    "surprise": 0.60,
    "neutral": 0.50,
    "sad": 0.15,
    "fear": 0.10,
    "angry": 0.05,
    "disgust": 0.05,
}

# How each emotion impacts different scenarios
EMOTION_SCENARIO_IMPACT = {
    "job_interview": {
        "happy": 0.85,
        "neutral": 0.70,
        "surprise": 0.40,
        "sad": 0.10,
        "angry": 0.05,
        "fear": 0.15,
        "disgust": 0.05,
    },
    "business_deal": {
        "happy": 0.80,
        "neutral": 0.75,
        "surprise": 0.35,
        "sad": 0.10,
        "angry": 0.10,
        "fear": 0.10,
        "disgust": 0.05,
    },
    "date": {
        "happy": 0.95,
        "surprise": 0.60,
        "neutral": 0.40,
        "sad": 0.10,
        "angry": 0.02,
        "fear": 0.10,
        "disgust": 0.02,
    },
}


class FacialExpressionAnalyzer:
    """Analyzes facial expressions across video frames to detect emotions."""

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.frame_emotions = []
        self.emotion_timeline = []

    def analyze_frame(self, frame: np.ndarray, frame_idx: int) -> dict | None:
        """
        Analyze a single frame for facial expressions.

        Returns:
            Dict with dominant emotion and all emotion probabilities, or None.
        """
        try:
            results = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            if isinstance(results, list):
                results = results[0]

            dominant = results.get("dominant_emotion", "neutral")
            emotions = results.get("emotion", {})

            result = {
                "frame_idx": frame_idx,
                "dominant_emotion": dominant,
                "emotions": emotions,
                "face_detected": results.get("face_confidence", 0) > self.confidence_threshold
                if "face_confidence" in results
                else True,
            }

            self.frame_emotions.append(result)
            self.emotion_timeline.append((frame_idx, dominant))
            return result

        except Exception as e:
            return {
                "frame_idx": frame_idx,
                "dominant_emotion": "unknown",
                "emotions": {},
                "face_detected": False,
                "error": str(e),
            }

    def analyze_frames(self, frames: list[tuple[int, np.ndarray]]) -> list[dict]:
        """Analyze multiple frames."""
        results = []
        for idx, frame in frames:
            result = self.analyze_frame(frame, idx)
            if result:
                results.append(result)
        return results

    def get_emotion_distribution(self) -> dict:
        """Get the overall distribution of emotions across all analyzed frames."""
        if not self.frame_emotions:
            return {}

        emotion_counts = {}
        total = 0
        for fe in self.frame_emotions:
            if fe.get("face_detected") and fe["dominant_emotion"] != "unknown":
                emo = fe["dominant_emotion"]
                emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
                total += 1

        if total == 0:
            return {"neutral": 1.0}

        return {emo: count / total for emo, count in emotion_counts.items()}

    def get_average_emotion_scores(self) -> dict:
        """Get average probability for each emotion across all frames."""
        if not self.frame_emotions:
            return {}

        all_emotions = {}
        count = 0
        for fe in self.frame_emotions:
            if fe.get("emotions"):
                count += 1
                for emo, score in fe["emotions"].items():
                    all_emotions[emo] = all_emotions.get(emo, 0) + score

        if count == 0:
            return {}

        return {emo: score / count for emo, score in all_emotions.items()}

    def get_facial_score(self) -> dict:
        """
        Compute overall facial expression scores for each scenario.

        Returns:
            Dict with scores for job_interview, business_deal, date (0-100).
        """
        distribution = self.get_emotion_distribution()
        if not distribution:
            return {"job_interview": 50.0, "business_deal": 50.0, "date": 50.0}

        scores = {}
        for scenario, impacts in EMOTION_SCENARIO_IMPACT.items():
            score = 0.0
            for emotion, proportion in distribution.items():
                impact = impacts.get(emotion, 0.3)
                score += proportion * impact
            scores[scenario] = round(min(score * 100, 100), 2)

        return scores

    def get_smile_ratio(self) -> float:
        """What fraction of frames had a happy/smiling expression."""
        if not self.frame_emotions:
            return 0.0
        happy_count = sum(
            1 for fe in self.frame_emotions
            if fe.get("dominant_emotion") == "happy" and fe.get("face_detected")
        )
        valid_frames = sum(1 for fe in self.frame_emotions if fe.get("face_detected"))
        return happy_count / valid_frames if valid_frames > 0 else 0.0

    def get_summary(self) -> dict:
        """Get a full summary of facial expression analysis."""
        return {
            "total_frames_analyzed": len(self.frame_emotions),
            "faces_detected": sum(1 for fe in self.frame_emotions if fe.get("face_detected")),
            "emotion_distribution": self.get_emotion_distribution(),
            "average_emotion_scores": self.get_average_emotion_scores(),
            "smile_ratio": round(self.get_smile_ratio(), 3),
            "scenario_scores": self.get_facial_score(),
            "emotion_timeline": self.emotion_timeline[:20],  # First 20 for brevity
        }
