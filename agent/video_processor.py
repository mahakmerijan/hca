"""
Video Processor Module
Handles extracting frames and audio from the input video.
"""

import cv2
import os
import numpy as np
try:
    from moviepy import VideoFileClip  # moviepy v2.x
except ImportError:
    from moviepy.editor import VideoFileClip  # moviepy v1.x fallback


class VideoProcessor:
    """Extracts frames and audio from video for downstream analysis."""

    def __init__(self, video_path: str, frame_sample_rate: int = 30):
        """
        Args:
            video_path: Path to the input video file.
            frame_sample_rate: Sample every Nth frame (e.g., 30 = ~1 frame/sec at 30fps).
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.video_path = video_path
        self.frame_sample_rate = frame_sample_rate
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_video_info(self) -> dict:
        """Return basic video metadata."""
        return {
            "path": self.video_path,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_seconds": round(self.duration, 2),
            "resolution": f"{self.width}x{self.height}",
            "frame_sample_rate": self.frame_sample_rate,
        }

    def extract_frames(self) -> list[tuple[int, np.ndarray]]:
        """
        Extract sampled frames from the video.

        Returns:
            List of (frame_index, frame_image) tuples.
        """
        return list(self.iter_frames())

    def get_sampled_frame_count(self) -> int:
        """Return the estimated number of sampled frames for this video."""
        if self.frame_sample_rate <= 0:
            return self.total_frames
        return max(1, (self.total_frames + self.frame_sample_rate - 1) // self.frame_sample_rate)

    def iter_frames(self):
        """
        Lazily iterate sampled frames from the video.

        Yields:
            Tuples of (frame_index, frame_image).
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_idx % self.frame_sample_rate == 0:
                yield frame_idx, frame
            frame_idx += 1

    def extract_audio(self, output_path: str = "output/audio.wav") -> str | None:
        """
        Extract audio track from video as WAV file.

        Args:
            output_path: Where to save the extracted audio.

        Returns:
            Path to the extracted audio file, or None if no audio.
        """
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        try:
            clip = VideoFileClip(self.video_path)
            if clip.audio is None:
                print("[VideoProcessor] No audio track found in video.")
                clip.close()
                return None
            clip.audio.write_audiofile(output_path, logger=None)
            clip.close()
            return output_path
        except Exception as e:
            print(f"[VideoProcessor] Audio extraction failed: {e}")
            return None

    def save_frames(self, frames: list[tuple[int, np.ndarray]], output_dir: str = "output/frames"):
        """Save extracted frames to disk."""
        os.makedirs(output_dir, exist_ok=True)
        for idx, frame in frames:
            path = os.path.join(output_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(path, frame)

    def release(self):
        """Release video capture resources."""
        if self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        self.release()
