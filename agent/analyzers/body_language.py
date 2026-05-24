"""
Body Language Analyzer
Uses MediaPipe Pose (Tasks API) to analyze posture, gestures, and body language confidence.
"""

import math
import os
import urllib.request
import numpy as np
import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmark = mp.tasks.vision.PoseLandmark
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# MediaPipe model URLs
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def _download_model(url: str, save_path: str):
    """Download a model file if it doesn't exist."""
    if os.path.exists(save_path):
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"    [BodyLanguage] Downloading model to {save_path}...")
    urllib.request.urlretrieve(url, save_path)


class BodyLanguageAnalyzer:
    """Analyzes body language and posture from video frames using MediaPipe Tasks API."""

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.frame_results = []

        # Download models
        models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
        self.pose_model_path = os.path.join(models_dir, "pose_landmarker_lite.task")
        self.hand_model_path = os.path.join(models_dir, "hand_landmarker.task")

        _download_model(POSE_MODEL_URL, self.pose_model_path)
        _download_model(HAND_MODEL_URL, self.hand_model_path)

        # Create Pose Landmarker
        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.pose_model_path),
            running_mode=VisionRunningMode.IMAGE,
            min_pose_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold,
        )
        self.pose_landmarker = PoseLandmarker.create_from_options(pose_options)

        # Create Hand Landmarker
        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.hand_model_path),
            running_mode=VisionRunningMode.IMAGE,
            min_hand_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold,
            num_hands=2,
        )
        self.hand_landmarker = HandLandmarker.create_from_options(hand_options)

    def _calculate_angle(self, a, b, c) -> float:
        """Calculate angle at point b given three landmark points (x,y)."""
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return angle

    def _assess_posture(self, landmarks: list) -> dict:
        """Assess posture quality from pose landmarks."""
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        NOSE = 0
        LEFT_HIP = 23
        RIGHT_HIP = 24

        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        shoulder_alignment = max(0, 1.0 - shoulder_diff * 10)

        nose = landmarks[NOSE]
        mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        head_offset = abs(nose.x - mid_shoulder_x)
        head_uprightness = max(0, 1.0 - head_offset * 5)

        spine_score = max(0, 1.0 - abs(nose.x - mid_shoulder_x) * 4)

        left_hip = landmarks[LEFT_HIP]
        right_hip = landmarks[RIGHT_HIP]
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        hip_width = abs(left_hip.x - right_hip.x)
        openness = min(shoulder_width / (hip_width + 1e-8), 2.0) / 2.0

        return {
            "shoulder_alignment": round(shoulder_alignment, 3),
            "head_uprightness": round(head_uprightness, 3),
            "spine_straightness": round(spine_score, 3),
            "openness": round(openness, 3),
        }

    def _assess_hand_gestures(self, frame: np.ndarray) -> dict:
        """Analyze hand gestures and visibility."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.hand_landmarker.detect(mp_image)

        if not results.hand_landmarks:
            return {
                "hands_visible": 0,
                "gesture_activity": 0.0,
                "hands_open": False,
            }

        num_hands = len(results.hand_landmarks)
        total_spread = 0
        hands_open = False

        for hand_lm in results.hand_landmarks:
            wrist = hand_lm[0]
            for tip_idx in [4, 8, 12, 16, 20]:
                tip = hand_lm[tip_idx]
                dist = math.sqrt((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2)
                total_spread += dist

            index_tip = hand_lm[8]
            pinky_tip = hand_lm[20]
            finger_spread = math.sqrt(
                (index_tip.x - pinky_tip.x) ** 2 + (index_tip.y - pinky_tip.y) ** 2
            )
            if finger_spread > 0.08:
                hands_open = True

        gesture_activity = min(total_spread / (num_hands * 0.8 + 1e-8), 1.0)

        return {
            "hands_visible": num_hands,
            "gesture_activity": round(gesture_activity, 3),
            "hands_open": hands_open,
        }

    def _assess_confidence_signals(self, landmarks: list) -> dict:
        """Detect confidence-related body language signals."""
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        NOSE = 0
        LEFT_HIP = 23
        RIGHT_HIP = 24

        left_wrist = landmarks[LEFT_WRIST]
        right_wrist = landmarks[RIGHT_WRIST]
        left_elbow = landmarks[LEFT_ELBOW]
        right_elbow = landmarks[RIGHT_ELBOW]
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        nose = landmarks[NOSE]
        left_hip = landmarks[LEFT_HIP]
        right_hip = landmarks[RIGHT_HIP]

        # Check if wrists are crossed
        left_wrist_near_right = abs(left_wrist.x - right_shoulder.x) < 0.1
        right_wrist_near_left = abs(right_wrist.x - left_shoulder.x) < 0.1
        arms_crossed = left_wrist_near_right and right_wrist_near_left

        # Arm angles
        left_arm_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Lean direction
        hip_mid_x = (left_hip.x + right_hip.x) / 2
        lean_direction = "centered"
        lean_amount = nose.x - hip_mid_x
        if lean_amount > 0.05:
            lean_direction = "leaning_right"
        elif lean_amount < -0.05:
            lean_direction = "leaning_left"

        # Forward lean
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        forward_lean = max(0, shoulder_y - nose.y)

        confidence_score = 0.5
        if not arms_crossed:
            confidence_score += 0.15
        if forward_lean > 0.05:
            confidence_score += 0.10
        if lean_direction == "centered":
            confidence_score += 0.10
        avg_arm_angle = (left_arm_angle + right_arm_angle) / 2
        if 60 < avg_arm_angle < 160:
            confidence_score += 0.15

        return {
            "arms_crossed": arms_crossed,
            "lean_direction": lean_direction,
            "forward_lean": round(forward_lean, 3),
            "avg_arm_angle": round(avg_arm_angle, 2),
            "confidence_score": round(min(confidence_score, 1.0), 3),
        }

    def analyze_frame(self, frame: np.ndarray, frame_idx: int) -> dict:
        """Analyze body language in a single frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.pose_landmarker.detect(mp_image)

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            result = {
                "frame_idx": frame_idx,
                "pose_detected": False,
                "posture": {},
                "hand_gestures": self._assess_hand_gestures(frame),
                "confidence_signals": {},
            }
            self.frame_results.append(result)
            return result

        landmarks = results.pose_landmarks[0]  # First detected pose
        posture = self._assess_posture(landmarks)
        hand_gestures = self._assess_hand_gestures(frame)
        confidence_signals = self._assess_confidence_signals(landmarks)

        result = {
            "frame_idx": frame_idx,
            "pose_detected": True,
            "posture": posture,
            "hand_gestures": hand_gestures,
            "confidence_signals": confidence_signals,
        }
        self.frame_results.append(result)
        return result

    def analyze_frames(self, frames: list[tuple[int, np.ndarray]]) -> list[dict]:
        """Analyze body language across multiple frames."""
        results = []
        for idx, frame in frames:
            result = self.analyze_frame(frame, idx)
            results.append(result)
        return results

    def get_body_language_score(self) -> dict:
        """
        Compute overall body language scores for each scenario.

        Returns:
            Dict with scores for job_interview, business_deal, date (0-100).
        """
        if not self.frame_results:
            return {"job_interview": 50.0, "business_deal": 50.0, "date": 50.0}

        posture_scores = []
        confidence_scores = []
        gesture_scores = []
        openness_scores = []

        for fr in self.frame_results:
            if fr.get("pose_detected"):
                posture = fr.get("posture", {})
                posture_avg = np.mean([
                    posture.get("shoulder_alignment", 0.5),
                    posture.get("head_uprightness", 0.5),
                    posture.get("spine_straightness", 0.5),
                ])
                posture_scores.append(posture_avg)
                openness_scores.append(posture.get("openness", 0.5))

                conf = fr.get("confidence_signals", {})
                confidence_scores.append(conf.get("confidence_score", 0.5))

            gestures = fr.get("hand_gestures", {})
            gesture_scores.append(gestures.get("gesture_activity", 0.0))

        avg_posture = np.mean(posture_scores) if posture_scores else 0.5
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
        avg_gesture = np.mean(gesture_scores) if gesture_scores else 0.3
        avg_openness = np.mean(openness_scores) if openness_scores else 0.5

        job_score = (avg_posture * 0.30 + avg_confidence * 0.30 +
                     avg_gesture * 0.20 + avg_openness * 0.20) * 100
        business_score = (avg_posture * 0.25 + avg_confidence * 0.35 +
                          avg_gesture * 0.20 + avg_openness * 0.20) * 100
        date_score = (avg_posture * 0.20 + avg_confidence * 0.20 +
                      avg_gesture * 0.25 + avg_openness * 0.35) * 100

        return {
            "job_interview": round(min(float(job_score), 100), 2),
            "business_deal": round(min(float(business_score), 100), 2),
            "date": round(min(float(date_score), 100), 2),
        }

    def get_summary(self) -> dict:
        """Get full summary of body language analysis."""
        pose_detected_count = sum(1 for fr in self.frame_results if fr.get("pose_detected"))

        avg_metrics = {}
        if pose_detected_count > 0:
            all_posture = [fr["posture"] for fr in self.frame_results if fr.get("pose_detected")]
            avg_metrics["avg_shoulder_alignment"] = round(
                float(np.mean([p.get("shoulder_alignment", 0) for p in all_posture])), 3
            )
            avg_metrics["avg_head_uprightness"] = round(
                float(np.mean([p.get("head_uprightness", 0) for p in all_posture])), 3
            )
            avg_metrics["avg_openness"] = round(
                float(np.mean([p.get("openness", 0) for p in all_posture])), 3
            )

            all_conf = [fr["confidence_signals"] for fr in self.frame_results if fr.get("pose_detected")]
            avg_metrics["avg_confidence_score"] = round(
                float(np.mean([c.get("confidence_score", 0) for c in all_conf])), 3
            )
            arms_crossed_ratio = sum(
                1 for c in all_conf if c.get("arms_crossed")
            ) / len(all_conf)
            avg_metrics["arms_crossed_ratio"] = round(arms_crossed_ratio, 3)

        return {
            "total_frames_analyzed": len(self.frame_results),
            "pose_detected_count": pose_detected_count,
            "average_metrics": avg_metrics,
            "scenario_scores": self.get_body_language_score(),
        }

    def release(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'pose_landmarker'):
            self.pose_landmarker.close()
        if hasattr(self, 'hand_landmarker'):
            self.hand_landmarker.close()
