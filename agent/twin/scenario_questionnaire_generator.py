"""
ScenarioQuestionnaireGenerator
================================
When a user says they are preparing for a specific scenario or person,
this generates 20-25 targeted follow-up questions to understand their context
more deeply, so we can build a better custom persona for simulation.
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

_SYSTEM_PROMPT = """You are a preparation coach helping someone get ready for an important interaction.
Given a brief description of what they are preparing for, generate a set of targeted questions
to understand their situation more deeply so that a realistic simulation partner can be created.

Your questions must:
1. Be specific to the scenario/person they described — not generic
2. Uncover the stakes, the relationship dynamics, the goals, and the likely challenges
3. Include a mix of question types: open-ended reflection, scale (1-10), and multiple-choice
4. Feel natural and conversational, not like a survey

Return ONLY a JSON array of question objects. Each object must have:
{
  "id": "q1",            // unique id
  "label": "...",        // short header
  "question": "...",     // the full question text
  "type": "textarea | scale | single_select | multi_select | text",
  "options": [...],      // only for single_select or multi_select
  "scale_min": 1,        // only for scale type
  "scale_max": 10,       // only for scale type
  "scale_label_min": "...",   // label for min
  "scale_label_max": "...",   // label for max
  "placeholder": "...",  // hint text for textarea/text
  "required": true
}

Generate at least 20 questions — aim for 20-25. Cover the scenario thoroughly:
- Background & context (relationship, history, stakes)
- Goals & desired outcomes
- The other party (personality, communication style, known objections)
- Your strengths, weaknesses, and concerns
- Tactical preparation (key messages, questions you expect, red lines)
- Emotional readiness and confidence
- Logistics and constraints
No preamble, no markdown — pure JSON array."""


class ScenarioQuestionnaireGenerator:
    def __init__(self):
        self.model_name = os.getenv("LLM_MODEL", "gemini-2.5-pro")
        self._project = os.getenv("VERTEX_PROJECT", "ai-ml-integrations")
        self._location = os.getenv("VERTEX_LOCATION", "us-central1")
        self.client = None

        if GENAI_AVAILABLE:
            self.client = genai.Client(
                vertexai=True,
                project=self._project,
                location=self._location,
            )

    def generate_questions(self, scenario_description: str, user_id: Optional[str] = None) -> list:
        """
        Given the user's description of what they're preparing for,
        return a list of targeted follow-up question objects.
        """
        if not self.client:
            return self._fallback_questions(scenario_description)

        user_prompt = (
            f"The user is preparing for the following scenario or interaction:\n\n"
            f"\"{scenario_description}\"\n\n"
            f"Generate at least 20 targeted questions (aim for 20-25) to help us understand "
            f"their situation in depth so we can build a realistic simulation partner for them."
        )

        try:
            from agent.token_logger import log_call, extract_token_counts
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config={
                    "system_instruction": _SYSTEM_PROMPT,
                    "temperature": 0.7,
                    "max_output_tokens": 8192,
                },
            )
            _inp, _out = extract_token_counts(response)
            log_call(self.model_name, "ScenarioQuestionnaireGenerator", _inp, _out,
                     user_id=user_id, extra={"scenario": scenario_description[:120]})

            text = (response.text or "").strip()
            # Strip markdown fences
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3].strip()
            if text.startswith("json"):
                text = text[4:].strip()

            questions = json.loads(text)
            if not isinstance(questions, list):
                raise ValueError("Expected a JSON array")
            return questions

        except Exception as e:
            print(f"[ScenarioQuestionnaireGenerator] Error: {e}")
            return self._fallback_questions(scenario_description)

    def _fallback_questions(self, description: str) -> list:
        """Generic fallback questions if LLM is unavailable."""
        return [
            {
                "id": "sq_relationship",
                "label": "Relationship",
                "question": "What is your current relationship with this person / organisation?",
                "type": "textarea",
                "placeholder": "e.g. I have met them once at a conference, or this is a cold outreach…",
                "required": True,
            },
            {
                "id": "sq_goal",
                "label": "Your Goal",
                "question": "What is the single most important outcome you want from this interaction?",
                "type": "textarea",
                "placeholder": "e.g. I want them to agree to a second meeting / offer me the position",
                "required": True,
            },
            {
                "id": "sq_stakes",
                "label": "Stakes",
                "question": "How important is this interaction to you right now?",
                "type": "scale",
                "scale_min": 1,
                "scale_max": 10,
                "scale_label_min": "Nice to have",
                "scale_label_max": "Career-defining",
                "required": True,
            },
            {
                "id": "sq_biggest_concern",
                "label": "Biggest Concern",
                "question": "What aspect of this conversation are you most worried about?",
                "type": "textarea",
                "placeholder": "e.g. Being asked about a gap in my CV / not having the right answers",
                "required": True,
            },
            {
                "id": "sq_person_style",
                "label": "Their Communication Style",
                "question": "Based on what you know, how would you describe the other person's communication style?",
                "type": "single_select",
                "options": [
                    "Very formal and direct",
                    "Analytical — data-driven, lots of questions",
                    "Warm and relationship-focused",
                    "Aggressive / high-pressure",
                    "Laid-back and casual",
                    "I don't know much about them yet",
                ],
                "required": False,
            },
            {
                "id": "sq_past_attempts",
                "label": "Past Attempts",
                "question": "Have you had a similar conversation before? What happened?",
                "type": "textarea",
                "placeholder": "e.g. I tried this last year and it didn't go well because…",
                "required": False,
            },
            {
                "id": "sq_strengths_here",
                "label": "Your Strengths Here",
                "question": "What do you think you bring to this conversation that could win them over?",
                "type": "textarea",
                "placeholder": "e.g. deep expertise, 5 years of relevant experience, unique insight…",
                "required": True,
            },
            {
                "id": "sq_hard_questions",
                "label": "Hardest Question",
                "question": "What is the hardest question they could ask you — and what would you say?",
                "type": "textarea",
                "placeholder": "Their question: …   My current answer: …",
                "required": False,
            },
        ]
