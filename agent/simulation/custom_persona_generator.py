"""
CustomPersonaGenerator
=======================
Creates a realistic counter-party persona from:
  - user's free-text description of the person/scenario
  - user's answers to the scenario-specific questionnaire

The resulting persona has the same shape as the built-in archetypes
used by counter_agents.py so it can drop straight into the simulation.
"""

import json
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

_SYSTEM_PROMPT = """You are a character designer for a social simulation platform.
Given a description of a real (or hypothetical) person/situation that a user is preparing to interact with,
create a detailed simulation persona that the user can practice against.

The persona must be:
- Realistic and nuanced — not a caricature
- Grounded in the details the user has provided
- Challenging but fair — it should stress-test the user, not destroy them

Return ONLY a JSON object with this exact schema:
{
  "persona_name": "...",                   // first name or role name
  "persona_title": "...",                  // e.g. "Series A Investor", "Senior Recruiter at Google"
  "personality_summary": "...",            // 2-3 sentence description of this person
  "communication_style": "...",            // how they speak and respond
  "primary_goal": "...",                   // what they want from this interaction
  "secondary_goal": "...",                 // underlying concern or hidden agenda
  "openness_to_candidate": <1-10>,         // 1=hostile/skeptical, 10=open/enthusiastic
  "pressure_level": <1-10>,               // how much they push back / challenge
  "system_prompt": "...",                  // full persona instruction for the LLM to embody this character (200-400 words)
  "opening_line": "...",                   // how they open the conversation
  "curveball_questions": [                 // 3-5 tough questions they will ask
    "...",
    "..."
  ],
  "red_flags_that_lose_them": [           // 3-5 things that would make them disengage
    "...",
    "..."
  ],
  "what_wins_them_over": [               // 3-5 things that would impress them
    "...",
    "..."
  ],
  "category": "job_interview | investor_pitch | dating | negotiation | general"
}

No preamble, no markdown, no trailing text — pure JSON object."""


class CustomPersonaGenerator:
    def __init__(self):
        self.model_name = os.getenv("LLM_MODEL", "gemini-3.5-flash")
        self._project = os.getenv("VERTEX_PROJECT", "ai-ml-integrations")
        self._location = os.getenv("VERTEX_LOCATION", "us-central1")
        self.client = None

        if GENAI_AVAILABLE:
            self.client = genai.Client(
                vertexai=True,
                project=self._project,
                location=self._location,
            )

    def generate(
        self,
        description: str,
        questionnaire_answers: Optional[dict] = None,
        user_id: Optional[str] = None,
        user_twin_summary: Optional[str] = None,
    ) -> dict:
        """
        Build a custom counter-party persona.

        Args:
            description:           User's free-text description of who they're preparing for
            questionnaire_answers: Full twinFormData dict (general + sq_* scenario answers)
            user_id:               Optional user context for token logging
            user_twin_summary:     Brief persona summary of the user's own twin (for calibration)
        """
        if not self.client:
            return self._fallback_persona(description)

        # Separate general personality answers from scenario-specific (sq_*) answers
        general_answers, scenario_answers = {}, {}
        for qid, answer in (questionnaire_answers or {}).items():
            if qid.startswith("sq_") or qid.startswith("_scenario"):
                scenario_answers[qid] = answer
            else:
                general_answers[qid] = answer

        def _fmt(d):
            return "\n".join(f"  - {k}: {v}" for k, v in d.items() if v not in (None, "", [], {}))

        user_context = ""
        if user_twin_summary:
            user_context += f"\n\n=== WHO THEY'RE MEETING (the user's own profile) ===\n{user_twin_summary}\n"
        if general_answers:
            user_context += f"\n\n=== USER'S GENERAL PERSONALITY (context for calibration) ===\n{_fmt(general_answers)}\n"
        if scenario_answers:
            user_context += f"\n\n=== USER'S SCENARIO-SPECIFIC ANSWERS (about this meeting) ===\n{_fmt(scenario_answers)}\n"

        user_prompt = (
            f"The user is preparing for the following interaction:\n\n"
            f"\"{description}\"\n"
            f"{user_context}\n\n"
            f"Using ALL the above context, create a simulation persona that realistically represents "
            f"the person/situation described. Calibrate their pressure level, communication style, "
            f"and curveball questions specifically to challenge THIS user based on their known profile."
        )

        try:
            from agent.token_logger import log_call, extract_token_counts
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config={
                    "system_instruction": _SYSTEM_PROMPT,
                    "temperature": 0.75,
                    "max_output_tokens": 4096,
                },
            )
            _inp, _out = extract_token_counts(response)
            log_call(self.model_name, "CustomPersonaGenerator", _inp, _out,
                     user_id=user_id, extra={"description": description[:120]})

            text = (response.text or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3].strip()
            if text.startswith("json"):
                text = text[4:].strip()

            # Handle thinking-model wrapping
            import re
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                text = m.group(0)

            persona = json.loads(text)
            persona["_source_description"] = description
            return persona

        except Exception as e:
            print(f"[CustomPersonaGenerator] Error: {e}")
            return self._fallback_persona(description)

    def _fallback_persona(self, description: str) -> dict:
        return {
            "persona_name": "Alex",
            "persona_title": "Interviewer",
            "personality_summary": (
                f"A professional evaluating the candidate for: {description[:200]}. "
                "They are analytical and expect concrete answers."
            ),
            "communication_style": "Direct and professional",
            "primary_goal": "Assess whether the candidate is the right fit",
            "secondary_goal": "Identify any red flags quickly",
            "openness_to_candidate": 6,
            "pressure_level": 6,
            "system_prompt": (
                f"You are Alex, a professional evaluating someone for: {description}. "
                "Ask probing but fair questions. Push back on vague answers. "
                "Be direct, professional, and keep the conversation focused on results and evidence."
            ),
            "opening_line": "Thanks for making the time. Let's get started — tell me a bit about yourself.",
            "curveball_questions": [
                "Tell me about a time you failed. What did you learn?",
                "Why should I choose you over other candidates?",
                "What's your biggest professional weakness right now?",
            ],
            "red_flags_that_lose_them": [
                "Vague or evasive answers",
                "No concrete examples",
                "Arrogance without substance",
            ],
            "what_wins_them_over": [
                "Specific, results-driven examples",
                "Honest self-awareness",
                "Genuine curiosity and engagement",
            ],
            "category": "general",
            "_source_description": description,
        }
