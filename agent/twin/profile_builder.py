"""
Twin Profile Builder
====================
Combines form answers with video analysis data to build a
structured Digital Twin profile JSON.
"""

import json
from typing import Optional


class TwinProfileBuilder:
    """
    Builds a structured persona profile from:
      1. Form responses (behavioral + cognitive sections)
      2. Video/audio analysis results (embodied model)
    """

    def __init__(self):
        pass

    def build(
        self,
        form_data: dict,
        video_analysis: Optional[dict] = None,
        user_name: str = "User"
    ) -> dict:
        """
        Merge form data + video analysis into a unified twin profile.

        Returns a profile dict ready to be fed to PersonaGenerator.
        """
        profile = {
            "name": user_name,
            "behavioral_model": self._build_behavioral(form_data),
            "cognitive_model": self._build_cognitive(form_data),
            "embodied_model": self._build_embodied(form_data, video_analysis),
        }
        return profile

    # ── Private builders ──────────────────────────────────────────

    def _build_behavioral(self, form: dict) -> dict:
        return {
            "introvert_extrovert_score": form.get("introvert_extrovert", 5),
            "risk_taking_score": form.get("risk_taking", 5),
            "personality_traits": form.get("personality_traits", []),
            "communication_style": form.get("communication_style", ""),
            "decision_style": form.get("decision_style", ""),
            "social_energy": form.get("social_energy", ""),
            "conflict_response": form.get("conflict_response", ""),
            "daily_habits": form.get("daily_habits", []),
            "morning_routine": form.get("morning_routine", ""),
            # Rich narrative fields
            "biggest_strength":   form.get("biggest_strength", ""),
            "biggest_weakness":   form.get("biggest_weakness", ""),
            "how_others_see_you": form.get("how_others_see_you", ""),
            "past_failure":       form.get("past_failure", ""),
        }

    def _build_cognitive(self, form: dict) -> dict:
        return {
            "career_goal": form.get("career_goal", ""),
            "dating_preferences": form.get("dating_preference", []),
            "business_risk_tolerance": form.get("business_risk_tolerance", ""),
            "core_values": form.get("core_values", []),
            "long_term_vision": form.get("long_term_goal", ""),
            "investment_style": form.get("investment_style", ""),
            "stress_response": form.get("stress_response", ""),
            "learning_style": form.get("learning_style", ""),
            "ideal_workplace": form.get("ideal_workplace", []),
            "negotiation_style": form.get("negotiation_style", ""),
            # Rich narrative fields
            "career_story":       form.get("career_story", ""),
            "pitch_yourself":     form.get("pitch_yourself", ""),
            "ideal_relationship": form.get("ideal_relationship", ""),
            "what_blocks_you":    form.get("what_blocks_you", ""),
        }

    def _build_embodied(self, form: dict, video: Optional[dict]) -> dict:
        # Self-report from form
        embodied = {
            "self_reported": {
                "eye_contact_score":      form.get("self_eye_contact", 5),
                "posture":                form.get("self_posture", ""),
                "gesture_frequency":      form.get("self_gestures", 5),
                "smile_frequency":        form.get("self_smile", 5),
                "voice_confidence":       form.get("self_voice_confidence", 5),
                "perceived_confidence":   form.get("self_perceived_confidence", 5),
                "nervous_tells":          form.get("nervous_tells", ""),
                "confident_tells":        form.get("confident_tells", ""),
                "first_impression":       form.get("first_impression", ""),
            }
        }

        # Override/augment with actual video analysis if available
        if video:
            bl    = video.get("body_language_analysis", {})
            fa    = video.get("facial_analysis", {})
            vs    = video.get("voice_speech_analysis", {})
            am    = bl.get("average_metrics", {})
            audio = vs.get("audio_features", {})
            sc    = audio.get("speech_content", {})

            embodied["video_derived"] = {
                "spine_angle":           am.get("avg_head_uprightness"),
                "shoulder_alignment":    am.get("avg_shoulder_alignment"),
                "posture_openness":      am.get("avg_openness"),
                "arms_crossed_ratio":    am.get("arms_crossed_ratio"),
                "confidence_score":      am.get("avg_confidence_score"),
                "smile_ratio":           fa.get("smile_ratio"),
                "dominant_emotion":      _dominant_emotion(fa.get("emotion_distribution", {})),
                "emotion_distribution":  fa.get("emotion_distribution", {}),
                "emotion_timeline":      fa.get("emotion_timeline", []),
                "speaking_pace":         audio.get("speaking_rate", {}).get("pace_label"),
                "mean_pitch_hz":         audio.get("pitch", {}).get("mean_pitch"),
                "pitch_stability":       audio.get("pitch", {}).get("pitch_stability"),
                "filler_word_ratio":     sc.get("filler_word_ratio"),
                "vocabulary_richness":   sc.get("vocabulary_richness"),
                "confident_language_ratio": sc.get("confident_language_ratio"),
                "word_count":            sc.get("word_count"),
                "full_transcript":       vs.get("full_transcript") or sc.get("transcript_preview", ""),
                "hands_visible":         bl.get("average_metrics", {}).get("avg_hands_visible"),
                "gesture_activity":      bl.get("average_metrics", {}).get("avg_gesture_activity"),
            }

            # Key expression moments — from counselling report's moment_analysis
            # The Gemini counsellor stores its output under key "counselling"
            gemini_report = (
                video.get("counselling")
                or video.get("gemini_counsellor")
                or video.get("ai_analysis")
                or video.get("counsellor_report")
            )
            if gemini_report:
                embodied["ai_coach_assessment"] = gemini_report

                # Extract moment_analysis as key_expression_moments
                moments = gemini_report.get("moment_analysis", [])
                if moments:
                    embodied["key_expression_moments"] = [
                        {
                            "timestamp":    m.get("timestamp"),
                            "emotion":      m.get("emotion_detected"),
                            "issue":        m.get("why_its_off"),
                            "what_was_said": m.get("what_was_said"),
                            "suggestion":   m.get("better_delivery"),
                        }
                        for m in moments[:10]
                    ]

                # Scenario success probabilities from the counsellor
                scenarios = gemini_report.get("scenarios", {})
                if scenarios:
                    embodied["success_probabilities"] = {
                        k: v.get("probability") for k, v in scenarios.items() if isinstance(v, dict)
                    }

                # Strengths and weaknesses narrative
                embodied["coach_strengths"]  = gemini_report.get("strengths", [])
                embodied["coach_weaknesses"] = gemini_report.get("weaknesses", [])
                embodied["coach_verdict"]    = gemini_report.get("coach_verdict", "")
                embodied["improvement_plan"] = gemini_report.get("improvement_plan", [])
                embodied["personality_snapshot"] = gemini_report.get("personality_snapshot", "")

            # Also pull scenario predictions from the computed model predictions
            final_pred = video.get("final_predictions", {})
            if final_pred and "success_probabilities" not in embodied:
                embodied["success_probabilities"] = {
                    k: v.get("probability") for k, v in final_pred.items() if isinstance(v, dict)
                }

        return embodied

    def to_json(self, profile: dict) -> str:
        return json.dumps(profile, indent=2, default=str)


# ── Helpers ───────────────────────────────────────────────────────

def _dominant_emotion(distribution: dict) -> str:
    if not distribution:
        return "neutral"
    return max(distribution, key=distribution.get)
