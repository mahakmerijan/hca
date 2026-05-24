"""
Feedback Generator
==================
Translates cluster analysis results into human-readable, scenario-specific
coaching feedback matching the style already used by the Gemini Counsellor.

Outputs:
  - "You interrupted too often" style insights
  - Which scenarios succeeded vs failed
  - For failures: why it failed and exactly how to improve
  - Comparison insights ("You're successful with Analytical types but fail 90% with Emotional")
"""

import json
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


FEEDBACK_SYSTEM = """\
You are a world-class communication coach delivering the final debrief after
10 social interaction simulations for a user's Digital Twin.

Your tone: Direct, empathetic, data-driven — like the best coach money can buy.
Write in second person ("you", "your").
Reference specific numbers and patterns from the data.

Return ONLY valid JSON:
{
  "executive_summary": "3-4 sentence powerful summary of all 10 simulations.",
  "headline_stats": {
    "overall_success_rate": <0-100>,
    "best_category": {"name": "...", "success_rate": <0-100>},
    "worst_category": {"name": "...", "success_rate": <0-100>},
    "total_simulations": <n>
  },
  "scenario_verdicts": [
    {
      "category": "job_interview | investor_pitch | dating",
      "title": "Human-readable title",
      "success_rate": <0-100>,
      "emoji": "💼 | 💰 | ❤️",
      "verdict": "One-line verdict",
      "top_strength": "What worked",
      "top_weakness": "What failed most",
      "success_story": "Description of a typical successful run",
      "failure_story": "Description of a typical failed run"
    }
  ],
  "failure_breakdown": [
    {
      "rank": 1,
      "pattern": "Short pattern name (e.g. 'Risk Deflection')",
      "frequency": "In X% of your failures...",
      "description": "Specific description of what happened",
      "exact_fix": "Precise drill or technique to fix this",
      "categories_affected": ["investor_pitch"]
    }
  ],
  "personality_insights": [
    {
      "personality_type": "Analytical",
      "success_rate": <0-100>,
      "insight": "Why you perform this way with this personality type",
      "strategy": "How to approach this personality type better"
    }
  ],
  "comparison_insight": "e.g. You're highly successful with Analytical personalities but fail 90% of the time with Emotional personalities.",
  "the_one_thing": "The single most important behavior change that would have the highest impact.",
  "30_day_plan": [
    {
      "week": 1,
      "focus": "What to work on",
      "daily_drill": "Specific daily practice"
    }
  ],
  "motivational_close": "Warm, encouraging 2-3 sentence close."
}
"""


class FeedbackGenerator:
    """
    Converts raw cluster analysis into user-facing coaching feedback.
    """

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY", "")
        self.model_name = os.getenv("LLM_MODEL", "gemini-2.5-pro")
        self.available = GENAI_AVAILABLE and bool(self.api_key)
        if self.available:
            self.client = genai.Client(api_key=self.api_key)

    def generate(self, cluster_analysis: dict, twin_persona: dict) -> dict:
        """
        Generate structured coaching feedback from cluster analysis.
        
        Args:
            cluster_analysis: Output from FailureClusterAnalyzer.analyze()
            twin_persona: The Digital Twin persona dict
        
        Returns:
            Structured feedback dict ready for frontend rendering
        """
        if not self.available:
            return self._fallback_feedback(cluster_analysis)

        user_prompt = (
            f"CLUSTER ANALYSIS RESULTS:\n{json.dumps(cluster_analysis, indent=2, default=str)[:4000]}\n\n"
            f"TWIN PERSONA SUMMARY:\n{twin_persona.get('persona_summary', '')}\n"
            f"Twin communication style: {json.dumps(twin_persona.get('communication_fingerprint', {}))}\n\n"
            "Generate the final coaching debrief."
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config={
                    "system_instruction": FEEDBACK_SYSTEM,
                    "temperature": 0.6,
                    "max_output_tokens": 2048,
                    "thinking_config": {"thinking_budget": 1024},
                },
            )
            text = response.text.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = text[:-3].strip()
            if text.startswith("json"):
                text = text[4:].strip()
            return json.loads(text)

        except Exception as e:
            print(f"[FeedbackGenerator] Error: {e}")
            return self._fallback_feedback(cluster_analysis)

    def _fallback_feedback(self, cluster_analysis: dict) -> dict:
        """Basic feedback when Gemini is unavailable."""
        overall = cluster_analysis.get("overall_success_rate", 0)
        total = cluster_analysis.get("total_simulations", 0)
        cat_results = cluster_analysis.get("category_results", {})

        # Find best/worst category
        sorted_cats = sorted(
            cat_results.items(),
            key=lambda x: x[1].get("success_rate", 0),
            reverse=True,
        )

        scenario_verdicts = []
        emoji_map = {"job_interview": "💼", "investor_pitch": "💰", "dating": "❤️"}
        title_map = {"job_interview": "Job Interviews", "investor_pitch": "Investor Pitches", "dating": "Dating"}
        for cat, data in cat_results.items():
            rate = data.get("success_rate", 0)
            scenario_verdicts.append({
                "category": cat,
                "title": title_map.get(cat, cat),
                "success_rate": rate,
                "emoji": emoji_map.get(cat, "🎯"),
                "verdict": "Strong performance" if rate >= 70 else "Needs improvement" if rate < 50 else "Average",
                "top_strength": "Consistency",
                "top_weakness": "Under pressure",
                "success_story": f"In successful runs you demonstrated confidence and clear communication.",
                "failure_story": f"In failed runs you showed hesitation under curveball questions.",
            })

        return {
            "executive_summary": (
                f"Your Digital Twin completed {total} simulations with a {overall}% overall success rate. "
                f"Enable Gemini API for deep coaching insights. "
                f"Your strongest area was {sorted_cats[0][0].replace('_', ' ') if sorted_cats else 'unknown'}."
            ),
            "headline_stats": {
                "overall_success_rate": overall,
                "best_category": {"name": sorted_cats[0][0] if sorted_cats else "N/A", "success_rate": sorted_cats[0][1].get("success_rate", 0) if sorted_cats else 0},
                "worst_category": {"name": sorted_cats[-1][0] if sorted_cats else "N/A", "success_rate": sorted_cats[-1][1].get("success_rate", 0) if sorted_cats else 0},
                "total_simulations": total,
            },
            "scenario_verdicts": scenario_verdicts,
            "failure_breakdown": [],
            "personality_insights": [],
            "comparison_insight": "Enable Gemini API for full personality comparison analysis.",
            "the_one_thing": "Focus on your weakest scenario first for maximum improvement.",
            "30_day_plan": [
                {"week": 1, "focus": "Identify core weakness", "daily_drill": "Practice answering curveball questions for 10 minutes daily"},
                {"week": 2, "focus": "Build confidence", "daily_drill": "Record yourself pitching and review body language"},
                {"week": 3, "focus": "Personality adaptation", "daily_drill": "Practice with a partner of the most challenging personality type"},
                {"week": 4, "focus": "Closing skills", "daily_drill": "Practice closing statements in front of mirror daily"},
            ],
            "motivational_close": f"You completed {total} simulations — that's more preparation than most people ever do. Keep pushing forward.",
        }
