"""
Referee Agent
=============
A separate Gemini instance that grades each micro-simulation interaction
on a 1-10 scale across three dimensions:
  - Alignment (did the twin's posture/tone match the target's preference?)
  - Friction   (where did the conversation stall?)
  - Outcome    (would this target "say yes"?)

As specified in architecture.md Step 3.
"""

import json
import os
import re
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


REFEREE_SYSTEM_PROMPT = """\
You are a neutral Referee Agent scoring a social simulation micro-interaction.

Your job is to evaluate a short 4-turn conversation between a "Digital Twin" 
(the user's AI replica) and a "Counter-Party" (recruiter, investor, or date).

Score on three dimensions (1-10 each) and provide brief reasoning.

Return ONLY valid JSON:
{
  "alignment_score": <1-10>,
  "alignment_reason": "Why this score — did the twin match the counter-party's preferences?",
  "friction_score": <1-10>,
  "friction_reason": "Where did the conversation stall or create resistance?",
  "outcome_score": <1-10>,
  "outcome_reason": "Would the counter-party 'say yes' — hire/invest/go on a second date?",
  "overall_score": <1-10>,
  "verdict": "success | partial_success | failure",
  "key_failure_moment": "The exact exchange where things went wrong, or null if success.",
  "key_success_moment": "The exact exchange that was the strongest, or null if failure.",
  "improvement_tags": ["tag1", "tag2"],
  "personality_alignment": "Which personality type did the twin perform best/worst with and why?"
}

Verdicts:
  success         = overall >= 7
  partial_success = overall >= 5
  failure         = overall < 5

improvement_tags must be from this list (pick 1-4 most relevant):
  low_confidence, poor_storytelling, lack_of_clarity, low_assertiveness,
  interrupted_too_much, too_passive, rambling_answers, weak_numbers,
  emotional_mismatch, poor_eye_contact_signals, filler_words,
  no_follow_up_questions, defensive_under_pressure, weak_opening,
  missed_closing_opportunity, overshared, undershared
"""


class RefereeAgent:
    """Grades each micro-simulation on alignment, friction, and outcome."""

    def __init__(self):
        # Use flash model — gemini-2.5-pro thinking tokens starve output budget
        self.model_name = os.getenv("SIM_LLM_MODEL", "gemini-2.5-flash")
        self._project = os.getenv("VERTEX_PROJECT", "ai-ml-integrations")
        self._location = os.getenv("VERTEX_LOCATION", "us-central1")
        self.available = GENAI_AVAILABLE
        if self.available:
            self.client = genai.Client(
                vertexai=True,
                project=self._project,
                location=self._location,
            )

    def grade(
        self,
        scenario: dict,
        conversation: list,
        twin_persona: dict,
    ) -> dict:
        """
        Grade a completed micro-simulation.

        Args:
            scenario: The scenario dict (category, counter_party, etc.)
            conversation: List of {"role": "twin"|"agent", "content": str} dicts
            twin_persona: The digital twin's persona dict
        """
        if not self.available:
            return self._fallback_grade(scenario)

        conversation_text = "\n".join(
            f"{turn['role'].upper()}: {turn['content']}"
            for turn in conversation
        )

        user_prompt = (
            f"Category: {scenario.get('category', 'unknown')}\n"
            f"Counter-party: {scenario['counter_party'].get('name', 'Agent')} "
            f"({scenario['counter_party'].get('personality', 'neutral')})\n"
            f"Counter-party goal: {scenario['counter_party'].get('goal', 'evaluate')}\n\n"
            f"Twin personality summary: {twin_persona.get('persona_summary', '')}\n\n"
            f"Full conversation:\n{conversation_text}\n\n"
            "Grade this interaction."
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config={
                    "system_instruction": REFEREE_SYSTEM_PROMPT,
                    "temperature": 0.3,
                    "max_output_tokens": 4096,
                    "thinking_config": {"thinking_budget": 0},
                },
            )
            raw = response.text
            if raw is None:
                # response.text is None when model returns only thought parts
                # iterate parts and pick the first non-thought text part
                try:
                    for part in response.candidates[0].content.parts:
                        if getattr(part, 'thought', False):
                            continue
                        if part.text:
                            raw = part.text
                            break
                    if raw is None:
                        finish_reason = getattr(response.candidates[0], 'finish_reason', 'unknown')
                        print(f"[RefereeAgent] No text part found. finish_reason={finish_reason}")
                        raw = ""
                except Exception as ex:
                    print(f"[RefereeAgent] Could not extract text from parts: {ex}")
                    raw = ""
            text = raw.strip()
            # strip markdown code fences
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = text[:-3].strip()
            if text.startswith("json"):
                text = text[4:].strip()
            # extract outermost JSON object — handles thinking tokens / prose wrapping
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                text = m.group(0)
            grade = json.loads(text)
            grade["scenario_id"] = scenario.get("scenario_id")
            grade["category"] = scenario.get("category")
            grade["counter_party_name"] = scenario["counter_party"].get("name", "")
            grade["counter_party_personality"] = scenario["counter_party"].get("personality", "")
            return grade
        except Exception as e:
            print(f"[RefereeAgent] Grade error: {e}")
            return self._fallback_grade(scenario)

    def _fallback_grade(self, scenario: dict) -> dict:
        return {
            "alignment_score": 5,
            "alignment_reason": "Unable to grade — Gemini unavailable",
            "friction_score": 5,
            "friction_reason": "Unable to grade",
            "outcome_score": 5,
            "outcome_reason": "Unable to grade",
            "overall_score": 5,
            "verdict": "partial_success",
            "key_failure_moment": None,
            "key_success_moment": None,
            "improvement_tags": [],
            "personality_alignment": "Unknown",
            "scenario_id": scenario.get("scenario_id"),
            "category": scenario.get("category"),
            "counter_party_name": scenario.get("counter_party", {}).get("name", ""),
            "counter_party_personality": scenario.get("counter_party", {}).get("personality", ""),
        }
