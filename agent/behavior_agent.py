"""
Behavior Analysis AI Agent
Orchestrates all analyzers and produces final compatibility predictions.
"""

import json
import os
import time
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text
from rich import box

from agent.video_processor import VideoProcessor
from agent.analyzers.facial_expression import FacialExpressionAnalyzer
from agent.analyzers.body_language import BodyLanguageAnalyzer
from agent.analyzers.voice_speech import VoiceSpeechAnalyzer


console = Console()


class BehaviorAnalysisAgent:
    """
    AI Agent that analyzes human behavior from video and predicts
    compatibility for job interviews, business deals, and dates.
    """

    def __init__(self, config: dict):
        self.config = config
        self.video_path = config.get("video_path", "video.mp4")
        analysis_cfg = config.get("analysis", {})
        weights_cfg = config.get("weights", {})

        self.frame_sample_rate = analysis_cfg.get("frame_sample_rate", 30)
        self.face_confidence = analysis_cfg.get("face_confidence_threshold", 0.5)
        self.pose_confidence = analysis_cfg.get("pose_confidence_threshold", 0.5)
        self.audio_segment_duration = analysis_cfg.get("audio_segment_duration", 5)

        self.weights = weights_cfg

        # Initialize components
        self.video_processor = None
        self.facial_analyzer = None
        self.body_analyzer = None
        self.voice_analyzer = None

        # Results storage
        self.results = {
            "video_info": {},
            "facial_analysis": {},
            "body_language_analysis": {},
            "voice_speech_analysis": {},
            "final_predictions": {},
            "behavioral_profile": {},
            "timestamp": "",
        }

    def _print_header(self):
        """Print a stylish header."""
        header = Text()
        header.append("🧠 ", style="bold")
        header.append("HUMAN BEHAVIOR ANALYSIS AGENT", style="bold cyan")
        header.append(" 🧠", style="bold")
        console.print(Panel(header, box=box.DOUBLE_EDGE, style="bright_cyan", padding=(1, 4)))
        console.print()

    def _print_step(self, step_num: int, total: int, description: str):
        """Print a step indicator."""
        console.print(
            f"  [bold yellow]▶ Step {step_num}/{total}:[/bold yellow] "
            f"[white]{description}[/white]"
        )

    def run(self) -> dict:
        """Run the full behavior analysis pipeline."""
        self._print_header()
        self.results["timestamp"] = datetime.now().isoformat()
        total_steps = 6

        # ─── Step 1: Load Video ───
        self._print_step(1, total_steps, "Loading and processing video...")
        try:
            self.video_processor = VideoProcessor(self.video_path, self.frame_sample_rate)
            self.results["video_info"] = self.video_processor.get_video_info()
            console.print(f"    [green]✓[/green] Video loaded: {self.results['video_info']['resolution']}, "
                          f"{self.results['video_info']['duration_seconds']}s, "
                          f"{self.results['video_info']['fps']:.1f} fps")
        except Exception as e:
            console.print(f"    [red]✗ Failed to load video: {e}[/red]")
            return self.results

        # ─── Step 2: Extract Frames ───
        self._print_step(2, total_steps, "Preparing video frames...")
        sampled_frame_count = self.video_processor.get_sampled_frame_count()
        console.print(f"    [green]✓[/green] Will analyze approximately {sampled_frame_count} sampled frames "
                      f"(1 every {self.frame_sample_rate} frames)")

        # ─── Step 3: Extract Audio ───
        self._print_step(3, total_steps, "Extracting audio track...")
        audio_path = self.video_processor.extract_audio()
        if audio_path:
            console.print(f"    [green]✓[/green] Audio extracted to {audio_path}")
        else:
            console.print("    [yellow]⚠[/yellow] No audio track found — voice analysis will be limited")

        # ─── Step 4: Facial Expression Analysis ───
        self._print_step(4, total_steps, "Analyzing facial expressions...")
        self.facial_analyzer = FacialExpressionAnalyzer(self.face_confidence)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("    Detecting emotions...", total=sampled_frame_count)
            for idx, frame in self.video_processor.iter_frames():
                self.facial_analyzer.analyze_frame(frame, idx)
                progress.advance(task)

        facial_summary = self.facial_analyzer.get_summary()
        self.results["facial_analysis"] = facial_summary
        console.print(f"    [green]✓[/green] Analyzed {facial_summary['faces_detected']} faces | "
                      f"Smile ratio: {facial_summary['smile_ratio']:.1%}")

        top_emotion = max(facial_summary["emotion_distribution"],
                          key=facial_summary["emotion_distribution"].get) if facial_summary["emotion_distribution"] else "N/A"
        console.print(f"    [green]✓[/green] Dominant emotion: [bold]{top_emotion}[/bold]")

        # ─── Step 5: Body Language Analysis ───
        self._print_step(5, total_steps, "Analyzing body language & posture...")
        self.body_analyzer = BodyLanguageAnalyzer(self.pose_confidence)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("    Analyzing posture...", total=sampled_frame_count)
            for idx, frame in self.video_processor.iter_frames():
                self.body_analyzer.analyze_frame(frame, idx)
                progress.advance(task)

        body_summary = self.body_analyzer.get_summary()
        self.results["body_language_analysis"] = body_summary
        console.print(f"    [green]✓[/green] Pose detected in {body_summary['pose_detected_count']} frames")

        avg_metrics = body_summary.get("average_metrics", {})
        if avg_metrics:
            console.print(f"    [green]✓[/green] Avg confidence: {avg_metrics.get('avg_confidence_score', 0):.2f} | "
                          f"Openness: {avg_metrics.get('avg_openness', 0):.2f}")

        # ─── Step 6: Voice & Speech Analysis ───
        self._print_step(6, total_steps, "Analyzing voice & speech patterns...")
        self.voice_analyzer = VoiceSpeechAnalyzer(self.audio_segment_duration)
        if audio_path:
            voice_features = self.voice_analyzer.run_full_analysis(audio_path)
            self.results["voice_speech_analysis"] = self.voice_analyzer.get_summary()

            rate = voice_features.get("speaking_rate", {})
            content = voice_features.get("speech_content", {})
            console.print(f"    [green]✓[/green] Speaking pace: {rate.get('pace_label', 'N/A')} "
                          f"({rate.get('estimated_syllables_per_sec', 0):.1f} syll/s)")
            console.print(f"    [green]✓[/green] Words transcribed: {content.get('word_count', 0)} | "
                          f"Filler ratio: {content.get('filler_word_ratio', 0):.1%}")
        else:
            self.results["voice_speech_analysis"] = self.voice_analyzer.get_summary()
            console.print("    [yellow]⚠[/yellow] Voice analysis skipped (no audio)")

        # ─── Compute Final Predictions ───
        console.print()
        console.print("  [bold magenta]🔮 Computing final predictions...[/bold magenta]")
        self._compute_final_predictions()
        self._build_behavioral_profile()
        self._display_results()

        # Save report
        output_cfg = self.config.get("output", {})
        if output_cfg.get("save_report", True):
            self._save_report(output_cfg.get("report_path", "output/analysis_report.txt"))

        # Cleanup
        if self.body_analyzer:
            self.body_analyzer.release()
        if self.video_processor:
            self.video_processor.release()

        return self.results

    def _compute_final_predictions(self):
        """Combine all analysis scores into final predictions."""
        facial_scores = self.results["facial_analysis"].get("scenario_scores", {})
        body_scores = self.results["body_language_analysis"].get("scenario_scores", {})
        voice_scores = self.results["voice_speech_analysis"].get("scenario_scores", {})

        predictions = {}
        for scenario in ["job_interview", "business_deal", "date"]:
            w = self.weights.get(scenario, {})

            facial_w = w.get("facial_expression", 0.25)
            body_w = w.get("body_language", 0.25)
            voice_w = w.get("voice_tone", 0.20)
            speech_w = w.get("speech_clarity", 0.15)
            conf_w = w.get("confidence_level", 0.15)

            facial_s = facial_scores.get(scenario, 50)
            body_s = body_scores.get(scenario, 50)
            voice_s = voice_scores.get(scenario, 50)

            # Confidence is derived from body language and voice
            confidence_s = (body_s * 0.6 + voice_s * 0.4)

            # Speech clarity from voice analyzer
            speech_s = voice_s  # Already includes clarity metrics

            final_score = (
                facial_s * facial_w +
                body_s * body_w +
                voice_s * voice_w +
                speech_s * speech_w +
                confidence_s * conf_w
            )

            # Clamp to 0-100
            final_score = max(0, min(100, final_score))

            predictions[scenario] = {
                "probability": round(final_score, 1),
                "grade": self._score_to_grade(final_score),
                "breakdown": {
                    "facial_expression": round(facial_s, 1),
                    "body_language": round(body_s, 1),
                    "voice_tone": round(voice_s, 1),
                    "speech_clarity": round(speech_s, 1),
                    "confidence": round(confidence_s, 1),
                },
            }

        self.results["final_predictions"] = predictions

    def _score_to_grade(self, score: float) -> str:
        """Convert a numeric score to a letter grade."""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B+"
        elif score >= 60:
            return "B"
        elif score >= 50:
            return "C+"
        elif score >= 40:
            return "C"
        elif score >= 30:
            return "D"
        else:
            return "F"

    def _frame_idx_to_timestamp(self, frame_idx: int) -> str:
        fps = float(self.results.get("video_info", {}).get("fps", 30) or 30)
        seconds = frame_idx / fps if fps > 0 else frame_idx
        total_seconds = int(round(seconds))
        minutes, secs = divmod(total_seconds, 60)
        return f"{minutes:02d}:{secs:02d}"

    def _build_behavioral_profile(self):
        """Build a human-readable behavioral profile."""
        predictions = self.results.get("final_predictions", {})
        facial = self.results.get("facial_analysis", {})
        body = self.results.get("body_language_analysis", {})
        voice = self.results.get("voice_speech_analysis", {})

        # Determine top strengths and areas for improvement
        strengths = []
        improvements = []

        # Facial
        smile_ratio = facial.get("smile_ratio", 0)
        if smile_ratio > 0.4:
            strengths.append("Warm and approachable facial expressions")
        elif smile_ratio < 0.1:
            improvements.append("Could benefit from more smiling and warmth")

        top_emotion = ""
        emo_dist = facial.get("emotion_distribution", {})
        if emo_dist:
            top_emotion = max(emo_dist, key=emo_dist.get)
            if top_emotion == "happy":
                strengths.append("Positive and upbeat emotional presence")
            elif top_emotion in ["angry", "sad", "fear"]:
                improvements.append(f"Facial expressions tend toward '{top_emotion}' — work on projecting positivity")

        # Body language
        avg_metrics = body.get("average_metrics", {})
        if avg_metrics:
            if avg_metrics.get("avg_confidence_score", 0) > 0.65:
                strengths.append("Strong confident body language")
            elif avg_metrics.get("avg_confidence_score", 0) < 0.4:
                improvements.append("Body language could project more confidence")

            if avg_metrics.get("avg_openness", 0) > 0.6:
                strengths.append("Open and welcoming posture")
            elif avg_metrics.get("avg_openness", 0) < 0.35:
                improvements.append("Posture appears closed — try more open body positioning")

            if avg_metrics.get("arms_crossed_ratio", 0) > 0.3:
                improvements.append("Arms crossed too frequently — signals defensiveness")

        # Voice
        voice_features = voice.get("audio_features", {})
        if voice_features:
            pace = voice_features.get("speaking_rate", {}).get("pace_label", "")
            if pace == "ideal":
                strengths.append("Excellent speaking pace")
            elif pace in ["very_slow", "very_fast"]:
                improvements.append(f"Speaking pace is {pace.replace('_', ' ')} — aim for moderate speed")

            content = voice_features.get("speech_content", {})
            if content.get("filler_word_ratio", 0) < 0.05:
                strengths.append("Clean speech with minimal filler words")
            elif content.get("filler_word_ratio", 0) > 0.15:
                improvements.append("Too many filler words (um, uh, like) — practice more polished delivery")

            if content.get("vocabulary_richness", 0) > 0.6:
                strengths.append("Rich and varied vocabulary")

        off_section_texts = []
        for seg in facial.get("off_segments", [])[:3]:
            off_section_texts.append(
                f"{self._frame_idx_to_timestamp(seg['frame_idx'])} — {seg['dominant_emotion']} ({seg['reason']})"
            )
        for seg in body.get("off_frames", [])[:3]:
            off_section_texts.append(
                f"{self._frame_idx_to_timestamp(seg['frame_idx'])} — {seg['reason']}"
            )
        # off_section_texts are stored in behavioral_profile["off_sections"]
        # and rendered in the dedicated UI card — do NOT add to improvements list

        self.results["behavioral_profile"] = {
            "dominant_emotion": top_emotion,
            "smile_ratio": smile_ratio,
            "strengths": strengths if strengths else ["Shows baseline competency across all areas"],
            "areas_for_improvement": improvements if improvements else ["No major areas of concern identified"],
            "off_sections": off_section_texts,
            "overall_impression": self._generate_impression(predictions),
        }

    def _generate_impression(self, predictions: dict) -> str:
        """Generate a brief overall impression."""
        avg_score = 0
        count = 0
        for scenario_data in predictions.values():
            avg_score += scenario_data.get("probability", 50)
            count += 1
        avg_score = avg_score / count if count > 0 else 50

        if avg_score >= 80:
            return "Excellent overall presence. This person communicates confidence, warmth, and competence effectively."
        elif avg_score >= 65:
            return "Good overall presence with notable strengths. Some refinement could elevate their impact further."
        elif avg_score >= 50:
            return "Average presentation with room for improvement. Targeted practice on key areas would help significantly."
        elif avg_score >= 35:
            return "Below average presence. Significant work needed on multiple dimensions of interpersonal communication."
        else:
            return "Weak overall presentation. Comprehensive coaching recommended across facial expressions, body language, and speech."

    def _display_results(self):
        """Display results in a beautiful terminal format."""
        console.print()
        predictions = self.results["final_predictions"]

        # ─── Main Predictions Table ───
        table = Table(
            title="🎯 COMPATIBILITY PREDICTIONS",
            box=box.HEAVY_EDGE,
            show_lines=True,
            title_style="bold white on blue",
            header_style="bold cyan",
            padding=(0, 2),
        )
        table.add_column("Scenario", style="bold white", width=22)
        table.add_column("Probability", justify="center", width=14)
        table.add_column("Grade", justify="center", width=8)
        table.add_column("Verdict", width=30)

        scenario_labels = {
            "job_interview": "💼 Cracking a Job",
            "business_deal": "🤝 Business Deal",
            "date": "❤️  Going on a Date",
        }

        for scenario, label in scenario_labels.items():
            data = predictions.get(scenario, {})
            prob = data.get("probability", 0)
            grade = data.get("grade", "N/A")

            # Color-code probability
            if prob >= 70:
                prob_style = "bold green"
                verdict = "Strong candidate!"
            elif prob >= 50:
                prob_style = "bold yellow"
                verdict = "Decent chance, room to grow"
            elif prob >= 35:
                prob_style = "bold bright_red"
                verdict = "Needs improvement"
            else:
                prob_style = "bold red"
                verdict = "Significant work needed"

            prob_str = f"[{prob_style}]{prob:.1f}%[/{prob_style}]"
            grade_color = "green" if grade.startswith("A") else "yellow" if grade.startswith("B") else "red"
            grade_str = f"[bold {grade_color}]{grade}[/bold {grade_color}]"

            table.add_row(label, prob_str, grade_str, verdict)

        console.print(table)

        # ─── Score Breakdown Table ───
        console.print()
        breakdown_table = Table(
            title="📊 SCORE BREAKDOWN",
            box=box.ROUNDED,
            show_lines=True,
            title_style="bold white on dark_green",
            header_style="bold",
        )
        breakdown_table.add_column("Component", style="bold", width=20)
        breakdown_table.add_column("💼 Job", justify="center", width=10)
        breakdown_table.add_column("🤝 Business", justify="center", width=10)
        breakdown_table.add_column("❤️  Date", justify="center", width=10)

        components = ["facial_expression", "body_language", "voice_tone", "speech_clarity", "confidence"]
        component_labels = {
            "facial_expression": "😊 Facial Expression",
            "body_language": "🧍 Body Language",
            "voice_tone": "🎤 Voice Tone",
            "speech_clarity": "💬 Speech Clarity",
            "confidence": "💪 Confidence",
        }

        for comp in components:
            row = [component_labels.get(comp, comp)]
            for scenario in ["job_interview", "business_deal", "date"]:
                val = predictions.get(scenario, {}).get("breakdown", {}).get(comp, 0)
                color = "green" if val >= 65 else "yellow" if val >= 45 else "red"
                row.append(f"[{color}]{val:.1f}[/{color}]")
            breakdown_table.add_row(*row)

        console.print(breakdown_table)

        # ─── Behavioral Profile ───
        profile = self.results.get("behavioral_profile", {})
        console.print()

        # Strengths
        strengths_text = "\n".join(f"  [green]✓[/green] {s}" for s in profile.get("strengths", []))
        console.print(Panel(
            strengths_text,
            title="[bold green]💪 STRENGTHS[/bold green]",
            box=box.ROUNDED,
            padding=(1, 2),
        ))

        # Areas for improvement
        improve_text = "\n".join(f"  [yellow]→[/yellow] {s}" for s in profile.get("areas_for_improvement", []))
        console.print(Panel(
            improve_text,
            title="[bold yellow]📈 AREAS FOR IMPROVEMENT[/bold yellow]",
            box=box.ROUNDED,
            padding=(1, 2),
        ))

        # Overall impression
        console.print(Panel(
            f"  {profile.get('overall_impression', '')}",
            title="[bold cyan]🧠 OVERALL IMPRESSION[/bold cyan]",
            box=box.DOUBLE_EDGE,
            style="bright_cyan",
            padding=(1, 2),
        ))
        console.print()

    def _save_report(self, report_path: str):
        """Save the full analysis report to a file."""
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("  HUMAN BEHAVIOR ANALYSIS REPORT\n")
            f.write(f"  Generated: {self.results['timestamp']}\n")
            f.write("=" * 70 + "\n\n")

            f.write("VIDEO INFORMATION\n")
            f.write("-" * 40 + "\n")
            for k, v in self.results["video_info"].items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

            f.write("FINAL PREDICTIONS\n")
            f.write("-" * 40 + "\n")
            scenario_labels = {
                "job_interview": "Cracking a Job Interview",
                "business_deal": "Getting a Business Deal",
                "date": "Going on a Date",
            }
            for scenario, label in scenario_labels.items():
                data = self.results["final_predictions"].get(scenario, {})
                f.write(f"\n  {label}:\n")
                f.write(f"    Probability: {data.get('probability', 0):.1f}%\n")
                f.write(f"    Grade: {data.get('grade', 'N/A')}\n")
                breakdown = data.get("breakdown", {})
                for comp, val in breakdown.items():
                    f.write(f"    - {comp}: {val:.1f}\n")
            f.write("\n")

            f.write("BEHAVIORAL PROFILE\n")
            f.write("-" * 40 + "\n")
            profile = self.results.get("behavioral_profile", {})
            f.write(f"  Dominant Emotion: {profile.get('dominant_emotion', 'N/A')}\n")
            f.write(f"  Smile Ratio: {profile.get('smile_ratio', 0):.1%}\n")
            f.write("\n  Strengths:\n")
            for s in profile.get("strengths", []):
                f.write(f"    ✓ {s}\n")
            f.write("\n  Areas for Improvement:\n")
            for s in profile.get("areas_for_improvement", []):
                f.write(f"    → {s}\n")
            f.write(f"\n  Overall Impression:\n    {profile.get('overall_impression', '')}\n")

            if profile.get("off_sections"):
                f.write("\n  Video Segments That Felt Off:\n")
                for off in profile.get("off_sections", []):
                    f.write(f"    - {off}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("  Full JSON data saved to: output/analysis_data.json\n")
            f.write("=" * 70 + "\n")

        # Also save raw JSON
        json_path = os.path.join(os.path.dirname(report_path), "analysis_data.json")
        with open(json_path, "w") as f:
            # Convert numpy types for JSON serialization
            json.dump(self.results, f, indent=2, default=str)

        console.print(f"  [dim]📄 Report saved to: {report_path}[/dim]")
        console.print(f"  [dim]📄 JSON data saved to: {json_path}[/dim]")
