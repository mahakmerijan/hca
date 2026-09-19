"""
LiveTwinQuestionGenerator
==========================
Phase 1 of the Live Digital Twin flow: the user describes the person they
are about to face (a recruiter, an investor, a date, a difficult client...).
This generates a short set of natural follow-up questions to understand
that person deeply enough to embody them convincingly in a live conversation.

Unlike ScenarioQuestionnaireGenerator (which asks the USER 20+ prep
questions about themselves), this focuses entirely on the OTHER PERSON —
but goes just as deep, generating at least 22 questions so the resulting
persona is as accurate and specific as possible.
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

_SYSTEM_PROMPT = """You are a persona-building assistant for a live conversation-practice app.
The user is about to have a real-time video conversation with an AI that will roleplay as
a specific person they are getting ready to face (e.g. a recruiter, an investor, a date, a
difficult boss, a client). Your job is to ask a thorough set of follow-up questions that will
let us build a convincing, deeply realistic AI persona of THAT PERSON.

Your questions must:
1. Be about the OTHER PERSON, not about the user
2. Be conversational and natural — this will be read aloud in a live chat, not a form
3. Be answerable in 1-3 sentences each — keep them light and quick
4. Cover the person thoroughly across ALL of these angles (spread questions across them):
   - Identity & role: who they are, their title/relationship, how long known, setting of the meeting
   - Personality: core traits, formality level, sense of humor, warmth vs. coldness
   - Communication style: pace, directness, how they ask questions, how they react when challenged
   - Attitude & mood: their likely attitude going in, current mood/context, stress level, priorities today
   - Values & motivations: what they care about most, what "success" looks like for them here
   - Likes & dislikes: what impresses them, what would turn them off or lose their trust
   - History & context: any prior interactions, shared history, known reputation
   - Pressure points: pet peeves, known hot buttons, topics to avoid or handle carefully
   - Decision-making style: analytical vs. intuitive, risk tolerance, how they push back
   - Non-verbal tendencies: how animated/expressive they tend to be, eye contact, energy level
   - Stakes for them: what they stand to gain or lose from this interaction
   - Wildcard: anything unusual, unpredictable, or specific to this exact person

Return ONLY a JSON array of AT LEAST 22 question objects (aim for 22-26 — do not return fewer
than 22). Each object must have:
{
  "id": "q1",
  "question": "...",
  "placeholder": "short example answer hint",
  "required": true
}
No preamble, no markdown — pure JSON array."""


class LiveTwinQuestionGenerator:
    def __init__(self):
        self.model_name = os.getenv("LLM_MODEL", "gemini-2.5-pro")
        self._project = os.getenv("VERTEX_PROJECT", "ai-ml-integrations")
        self._location = os.getenv("VERTEX_LOCATION", "us-central1")
        self.client = None
        if GENAI_AVAILABLE:
            self.client = genai.Client(vertexai=True, project=self._project, location=self._location)

    def generate_questions(self, description: str, user_id: Optional[str] = None, use_case: Optional[dict] = None) -> list:
        """Given the user's description of the person they'll face, return follow-up questions."""
        if not self.client:
            return self._fallback_questions()

        use_case_block = f"\n\nDOMAIN CONTEXT: {use_case['context_prompt']}" if use_case else ""
        user_prompt = (
            f"The user is about to have a live conversation with someone they describe as:\n\n"
            f"\"{description}\"\n\n"
            f"Generate AT LEAST 22 short follow-up questions (aim for 22-26) about THIS PERSON, "
            f"spread across identity, personality, communication style, attitude, values, likes/dislikes, "
            f"history, pressure points, decision-making style, non-verbal tendencies, and stakes — "
            f"so we can build a deeply accurate AI persona of them for the user to practice with."
            f"{use_case_block}"
        )
        try:
            from agent.token_logger import log_call, extract_token_counts
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config={
                    "system_instruction": _SYSTEM_PROMPT,
                    "temperature": 0.6,
                    "max_output_tokens": 8192,
                    # Minimal thinking budget — gemini-2.5-pro rejects 0, and unbounded
                    # thinking was silently eating the output budget, truncating questions.
                    "thinking_config": {"thinking_budget": 128},
                },
            )
            _inp, _out = extract_token_counts(response)
            log_call(self.model_name, "LiveTwinQuestionGenerator", _inp, _out,
                     user_id=user_id, extra={"description": description[:120]})

            text = (response.text or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3].strip()
            if text.startswith("json"):
                text = text[4:].strip()

            import re
            m = re.search(r'\[.*\]', text, re.DOTALL)
            if m:
                text = m.group(0)

            questions = json.loads(text)
            if not isinstance(questions, list) or len(questions) < 22:
                raise ValueError("Expected a JSON array of at least 22 questions")
            return questions
        except Exception as e:
            print(f"[LiveTwinQuestionGenerator] Error: {e}")
            return self._fallback_questions()

    def _fallback_questions(self) -> list:
        return [
            {"id": "q1", "question": "Who is this person and what's their role or relationship to you?",
             "placeholder": "e.g. Hiring manager at a mid-size startup", "required": True},
            {"id": "q2", "question": "How long have you known them, or is this your first interaction?",
             "placeholder": "e.g. First time meeting, only exchanged emails so far", "required": True},
            {"id": "q3", "question": "What's the setting — video call, in person, formal or casual?",
             "placeholder": "e.g. Formal video interview, 30 minutes", "required": True},
            {"id": "q4", "question": "How would you describe their core personality traits?",
             "placeholder": "e.g. Analytical, reserved, dry sense of humor", "required": True},
            {"id": "q5", "question": "How formal or casual are they typically?",
             "placeholder": "e.g. Fairly formal but relaxes once comfortable", "required": True},
            {"id": "q6", "question": "Do they have a sense of humor, and if so what kind?",
             "placeholder": "e.g. Dry, sarcastic, rarely jokes", "required": False},
            {"id": "q7", "question": "How would you describe their communication style — direct, roundabout, warm?",
             "placeholder": "e.g. Very direct, gets to the point fast", "required": True},
            {"id": "q8", "question": "How do they usually react when someone pushes back or challenges them?",
             "placeholder": "e.g. Gets more engaged, respects pushback if well-reasoned", "required": True},
            {"id": "q9", "question": "What do you think their attitude going into this will be?",
             "placeholder": "e.g. Skeptical at first, warms up if you're specific", "required": True},
            {"id": "q10", "question": "What's likely on their mind or their mood today, if you know?",
             "placeholder": "e.g. Busy quarter, may be short on time", "required": False},
            {"id": "q11", "question": "What does 'success' in this meeting look like from their side?",
             "placeholder": "e.g. Finding someone who can start immediately", "required": True},
            {"id": "q12", "question": "What do they seem to value most in general?",
             "placeholder": "e.g. Efficiency, honesty, proven results", "required": True},
            {"id": "q13", "question": "What would impress them?",
             "placeholder": "e.g. Concrete numbers and specific examples", "required": True},
            {"id": "q14", "question": "What would turn them off or lose their trust?",
             "placeholder": "e.g. Vagueness, over-promising", "required": True},
            {"id": "q15", "question": "Any history between you two, or context I should know?",
             "placeholder": "e.g. We've only emailed so far, never spoken", "required": False},
            {"id": "q16", "question": "Do they have a reputation you've heard about from others?",
             "placeholder": "e.g. Known for asking tough follow-up questions", "required": False},
            {"id": "q17", "question": "Any pet peeves or hot-button topics you should avoid or handle carefully?",
             "placeholder": "e.g. Hates being interrupted, sensitive about past layoffs", "required": False},
            {"id": "q18", "question": "Are they more analytical/data-driven or intuitive/gut-feel in how they decide things?",
             "placeholder": "e.g. Wants data before deciding anything", "required": True},
            {"id": "q19", "question": "How much risk are they comfortable with?",
             "placeholder": "e.g. Conservative, prefers proven approaches", "required": False},
            {"id": "q20", "question": "How animated or expressive are they typically — lots of energy, or more reserved?",
             "placeholder": "e.g. Reserved, doesn't show much emotion", "required": False},
            {"id": "q21", "question": "What do they stand to gain or lose from this interaction?",
             "placeholder": "e.g. Needs to fill this role before quarter end", "required": True},
            {"id": "q22", "question": "Anything unusual, unpredictable, or unique about this specific person?",
             "placeholder": "e.g. Likes to ask one totally unrelated question to see how you react", "required": False},
        ]
