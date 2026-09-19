#!/usr/bin/env python3
"""
Human Compatibility Analysis Agent
===================================
Analyzes a person's behavior from video and predicts their probability of:
  1. Cracking a Job Interview
  2. Getting a Business Deal
  3. Going on a Date

Usage:
    python main.py                          # Use default video.mp4
    python main.py --video path/to/video.mp4  # Specify video path
    python main.py --config custom_config.json # Use custom config
"""

import argparse
import json
import os
import sys

from rich.console import Console

from agent.behavior_agent import BehaviorAnalysisAgent

console = Console()


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        console.print(f"[yellow]⚠ Config file not found at {config_path}, using defaults.[/yellow]")
        return get_default_config()


def get_default_config() -> dict:
    """Return default configuration."""
    return {
        "video_path": "video.mp4",
        "analysis": {
            "frame_sample_rate": 30,
            "face_confidence_threshold": 0.5,
            "pose_confidence_threshold": 0.5,
            "audio_segment_duration": 5,
        },
        "weights": {
            "job_interview": {
                "facial_expression": 0.25,
                "body_language": 0.25,
                "voice_tone": 0.20,
                "speech_clarity": 0.15,
                "confidence_level": 0.15,
            },
            "business_deal": {
                "facial_expression": 0.20,
                "body_language": 0.20,
                "voice_tone": 0.25,
                "speech_clarity": 0.20,
                "confidence_level": 0.15,
            },
            "date": {
                "facial_expression": 0.30,
                "body_language": 0.25,
                "voice_tone": 0.20,
                "speech_clarity": 0.10,
                "confidence_level": 0.15,
            },
        },
        "output": {
            "save_report": True,
            "report_path": "output/analysis_report.txt",
            "save_annotated_frames": False,
            "annotated_frames_dir": "output/frames",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="🧠 Human Behavior Analysis AI Agent — Predict job, business, and date success from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="Path to the video file to analyze (default: video.mp4)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.json",
        help="Path to config JSON file (default: config.json)",
    )
    parser.add_argument(
        "--sample-rate", "-s",
        type=int,
        default=None,
        help="Frame sample rate — analyze every Nth frame (lower = more detailed but slower)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with CLI args
    if args.video:
        config["video_path"] = args.video
    if args.sample_rate:
        config.setdefault("analysis", {})["frame_sample_rate"] = args.sample_rate

    # Validate video exists
    if not os.path.exists(config["video_path"]):
        console.print(f"[bold red]✗ Error: Video file not found: {config['video_path']}[/bold red]")
        console.print("[dim]  Place your video as 'video.mp4' in the project directory, "
                      "or use --video to specify a path.[/dim]")
        sys.exit(1)

    # Run the agent
    agent = BehaviorAnalysisAgent(config)

    try:
        results = agent.run()
        console.print("[bold green]✅ Analysis complete![/bold green]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]✗ Analysis failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
