"""
LiveTwinPersonaBuilder
=======================
Phase 2 of the Live Digital Twin flow: builds a persona of the DESCRIBED
PERSON from the user's free-text description + their answers to the
follow-up questions, tuned for embodiment in a real-time spoken/video
conversation (short, natural turns — not essay-length replies).
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

_SYSTEM_PROMPT = """You are a character designer building an AI persona for a LIVE, real-time
video conversation-practice tool. The user described a real person they are about to face
(recruiter, investor, date, boss, client, etc.) and answered a few follow-up questions about
them. Build a persona that the AI will embody turn-by-turn while talking with the user live.

The persona must:
- Feel like a real individual, grounded in the details given — not a caricature
- Be realistic and fair, but still create authentic pressure appropriate to the situation
- Have a system_prompt written for LIVE spoken dialogue: short, natural turns (1-3 sentences),
  reacting in the moment to tone/confidence, never breaking character, never narrating actions

Return ONLY a JSON object with this exact schema:
{
  "persona_name": "...",
  "persona_role": "...",              // e.g. "Senior Recruiter", "First date", "Series A Investor"
  "personality_summary": "...",       // 2-3 sentences
  "communication_style": "...",       // how they talk: pace, tone, directness
  "primary_goal": "...",              // what they want from this interaction
  "openness_level": <1-10>,           // 1=guarded/skeptical, 10=warm/open
  "pressure_level": <1-10>,           // how much they challenge/push back
  "system_prompt": "...",             // full live-roleplay instruction, 150-300 words
  "opening_line": "...",              // first thing they say when the call starts
  "what_impresses_them": ["...", "..."],
  "what_turns_them_off": ["...", "..."],
  "likely_questions_they_will_ask": ["...", "...", "..."],
  "category": "job_interview | investor_pitch | dating | negotiation | difficult_conversation | general"
}
No preamble, no markdown, no trailing text — pure JSON object."""


class LiveTwinPersonaBuilder:
    def __init__(self):
        self.model_name = os.getenv("LLM_MODEL", "gemini-2.5-pro")
        self._project = os.getenv("VERTEX_PROJECT", "ai-ml-integrations")
        self._location = os.getenv("VERTEX_LOCATION", "us-central1")
        self.client = None
        if GENAI_AVAILABLE:
            self.client = genai.Client(vertexai=True, project=self._project, location=self._location)

    def build(
        self,
        description: str,
        answers: dict,
        user_id: Optional[str] = None,
        use_case: Optional[dict] = None,
        level: int = 2,
    ) -> dict:
        if not self.client:
            return self._fallback_persona(description, use_case=use_case, level=level)

        from agent.live.use_cases import get_twin_level
        twin_level = get_twin_level(level)

        answers_text = "\n".join(f"  - {q}: {a}" for q, a in (answers or {}).items() if a not in (None, ""))
        use_case_block = f"\n\nDOMAIN CONTEXT: {use_case['context_prompt']}" if use_case else ""
        user_prompt = (
            f"The user described the person they're about to face as:\n\n\"{description}\"\n\n"
            f"Follow-up answers about this person:\n{answers_text}\n\n"
            f"Build a live-roleplay persona of this exact person for the user to practice with."
            f"{use_case_block}\n\n"
            f"DIFFICULTY LEVEL — {twin_level['name']}: {twin_level['behavior']} "
            f"Calibrate openness_level around {twin_level['openness_level']}/10 and pressure_level around "
            f"{twin_level['pressure_level']}/10, and bake this difficulty behavior into the system_prompt."
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
                    # Minimal thinking budget — gemini-2.5-pro rejects 0, and unbounded
                    # thinking was silently eating the output budget, cutting sentences off mid-way.
                    "thinking_config": {"thinking_budget": 128},
                },
            )
            _inp, _out = extract_token_counts(response)
            log_call(self.model_name, "LiveTwinPersonaBuilder", _inp, _out,
                      user_id=user_id, extra={"description": description[:120]})

            text = (response.text or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3].strip()
            if text.startswith("json"):
                text = text[4:].strip()

            import re
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                text = m.group(0)

            persona = json.loads(text)
            persona["_source_description"] = description
            persona["_source_answers"] = answers
            persona["_use_case_id"] = (use_case or {}).get("id")
            persona["_level"] = level
            # Belt-and-suspenders: guarantee the difficulty behavior is present in the
            # system_prompt the live chat agent actually uses, even if the model forgot it.
            persona["system_prompt"] = (
                persona.get("system_prompt", "") + f"\n\n{twin_level['behavior']}"
            )
            return persona
        except Exception as e:
            print(f"[LiveTwinPersonaBuilder] Error: {e}")
            return self._fallback_persona(description, use_case=use_case, level=level)

    def _fallback_persona(self, description: str, use_case: Optional[dict] = None, level: int = 2) -> dict:
        from agent.live.use_cases import get_twin_level
        twin_level = get_twin_level(level)
        domain_note = f" Context: {use_case['context_prompt']}" if use_case else ""
        return {
            "persona_name": "Jordan",
            "persona_role": "The person you described",
            "personality_summary": f"A person evaluating this interaction: {description[:200]}",
            "communication_style": "Direct and observant",
            "primary_goal": "Assess whether this conversation is worth their time",
            "openness_level": twin_level["openness_level"],
            "pressure_level": twin_level["pressure_level"],
            "system_prompt": (
                f"You are Jordan, roleplaying as: {description}.{domain_note} Speak in short, natural, "
                "spoken-style turns (1-3 sentences). React authentically to what the user says — "
                "warm up if they're confident and specific, grow skeptical if they're vague. "
                "Never break character, never narrate actions, never say you are an AI."
                f"\n\n{twin_level['behavior']}"
            ),
            "opening_line": "Hey, thanks for making the time — let's get into it.",
            "what_impresses_them": ["Specific, concrete answers", "Genuine confidence"],
            "what_turns_them_off": ["Vagueness", "Over-explaining"],
            "likely_questions_they_will_ask": ["Tell me more about that.", "Why does this matter to you?"],
            "category": (use_case or {}).get("id", "general"),
            "_source_description": description,
            "_use_case_id": (use_case or {}).get("id"),
            "_level": level,
        }
