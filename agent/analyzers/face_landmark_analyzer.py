"""
Face Landmark Analyzer
Uses MediaPipe FaceLandmarker (Tasks API, blendshapes) to track eyebrow
movement and other fine-grained facial-muscle activity that DeepFace's
coarse emotion classifier does not expose.

Reuses the GLES/EGL headless stub already installed by body_language.py,
so this module must be imported after (or alongside) it.
"""

import logging
import os
import urllib.request

import numpy as np

# Importing body_language triggers `_preload_gles_stub()` as a side effect,
# and gives us its already-proven `_download_model` helper + mp/cv2 handles.
from agent.analyzers import body_language as _bl

mp = _bl.mp
cv2 = _bl.cv2
_MP_AVAILABLE = _bl._MP_AVAILABLE

logger = logging.getLogger(__name__)

FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

# Blendshape categories we care about for live coaching
BROW_KEYS = ["browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight"]
EXTRA_KEYS = ["jawOpen", "mouthSmileLeft", "mouthSmileRight", "eyeBlinkLeft", "eyeBlinkRight"]


class FaceLandmarkAnalyzer:
    """Tracks eyebrow movement + a few supporting facial blendshapes across frames."""

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.frame_results: list[dict] = []
        self.available = False
        self.error_message = None
        self.face_landmarker = None

        if not _MP_AVAILABLE:
            self.error_message = "mediapipe/cv2 not available"
            return

        models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
        self.model_path = os.path.join(models_dir, "face_landmarker.task")
        try:
            _bl._download_model(FACE_MODEL_URL, self.model_path)

            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            base_options = BaseOptions(model_asset_path=self.model_path, delegate=BaseOptions.Delegate.CPU)
            options = FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=VisionRunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=confidence_threshold,
                min_tracking_confidence=confidence_threshold,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
            )
            self.face_landmarker = FaceLandmarker.create_from_options(options)
            self.available = True
        except Exception as exc:
            self.available = False
            self.error_message = str(exc)
            logger.warning("[FaceLandmarkAnalyzer] MediaPipe FaceLandmarker unavailable: %s", exc)

    def analyze_frame(self, frame: np.ndarray, frame_idx: int) -> dict:
        """Analyze eyebrow/facial blendshape activity in a single frame."""
        if not self.available:
            result = {"frame_idx": frame_idx, "face_detected": False, "blendshapes": {}}
            self.frame_results.append(result)
            return result

        h, w = frame.shape[:2]
        if w > 640:
            scale = 640.0 / w
            frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            detection = self.face_landmarker.detect(mp_image)
        except Exception as e:
            result = {"frame_idx": frame_idx, "face_detected": False, "blendshapes": {}, "error": str(e)}
            self.frame_results.append(result)
            return result

        if not detection.face_blendshapes:
            result = {"frame_idx": frame_idx, "face_detected": False, "blendshapes": {}}
            self.frame_results.append(result)
            return result

        shapes = {c.category_name: c.score for c in detection.face_blendshapes[0]}
        result = {
            "frame_idx": frame_idx,
            "face_detected": True,
            "blendshapes": {k: round(float(shapes.get(k, 0.0)), 4) for k in (BROW_KEYS + EXTRA_KEYS)},
        }
        self.frame_results.append(result)
        return result

    def get_summary(self) -> dict:
        """Aggregate eyebrow-movement activity across all analyzed frames."""
        detected = [fr for fr in self.frame_results if fr.get("face_detected")]
        if not detected:
            summary = {
                "total_frames_analyzed": len(self.frame_results),
                "faces_detected": 0,
                "average_blendshapes": {},
                "eyebrow_movement_activity": 0.0,
                "eyebrow_raise_ratio": 0.0,
                "eyebrow_furrow_ratio": 0.0,
            }
            if not self.available:
                summary["analysis_mode"] = "fallback"
                summary["error"] = self.error_message
            return summary

        all_keys = BROW_KEYS + EXTRA_KEYS
        series = {k: np.array([fr["blendshapes"].get(k, 0.0) for fr in detected]) for k in all_keys}

        avg = {k: round(float(np.mean(v)), 4) for k, v in series.items()}
        # Movement activity = how much the brow blendshapes fluctuate frame-to-frame
        # (a mostly-frozen/monotone face scores low; expressive, reactive eyebrows score higher).
        brow_std = np.mean([float(np.std(series[k])) for k in BROW_KEYS])
        raise_ratio = float(np.mean((series["browOuterUpLeft"] + series["browOuterUpRight"]) / 2 > 0.3))
        furrow_ratio = float(np.mean((series["browDownLeft"] + series["browDownRight"]) / 2 > 0.3))

        return {
            "total_frames_analyzed": len(self.frame_results),
            "faces_detected": len(detected),
            "average_blendshapes": avg,
            "eyebrow_movement_activity": round(float(brow_std), 4),
            "eyebrow_raise_ratio": round(raise_ratio, 3),
            "eyebrow_furrow_ratio": round(furrow_ratio, 3),
        }

    def release(self):
        if self.face_landmarker:
            try:
                self.face_landmarker.close()
            except Exception:
                pass
