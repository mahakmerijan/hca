"""
Counter-Party Agents
====================
Gemini-powered agents that simulate the other side of a conversation:
  - RecruiterAgent     (job interviews)
  - InvestorAgent      (business pitches)
  - DateAgent          (dating scenarios)

Each agent has a unique personality, goals, and response style
derived from the scenario's counter-party archetype.
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

_client: Optional[object] = None


def _get_client():
    global _client
    if _client is None and GENAI_AVAILABLE:
        _client = genai.Client(
            vertexai=True,
            project=os.getenv("VERTEX_PROJECT", "ai-ml-integrations"),
            location=os.getenv("VERTEX_LOCATION", "us-central1"),
        )
    return _client


def _call_gemini(system: str, user: str, temperature: float = 0.8) -> str:
    """Shared Gemini call for all counter-party agents."""
    client = _get_client()
    # Use flash model for counter-party responses — fast, non-thinking, sufficient for dialogue
    model_name = os.getenv("SIM_LLM_MODEL", "gemini-2.0-flash")
    if client is None:
        return "[Gemini unavailable — using placeholder response]"
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=user,
            config={"system_instruction": system, "temperature": temperature, "max_output_tokens": 200},
        )
        text = response.text
        if text is None:
            try:
                text = response.candidates[0].content.parts[0].text
            except Exception:
                text = ""
        return (text or "").strip() or "[Agent is thinking...]"
    except Exception as e:
        return f"[Agent error: {e}]"


class RecruiterAgent:
    """
    Simulates a recruiter/hiring manager in a job interview.
    Personality driven by the scenario archetype.
    """

    def build_system_prompt(self, archetype: dict) -> str:
        return f"""You are '{archetype.get("name", "Recruiter")}' — a hiring manager conducting a job interview.

Your personality: {archetype.get("personality", "professional")}
Your goal today: {archetype.get("goal", "assess candidate fit")}
Your interview style: {archetype.get("style", "standard")}
Your emotional responsiveness: {archetype.get("emotional", "medium")}
Your patience level: {archetype.get("patience", "medium")}
Your mood today: {archetype.get("mood_today", "focused")}
Context: {archetype.get("prior_context", "Neutral")}

RULES:
1. Stay completely in character — never break the 4th wall
2. React authentically to what the candidate says
3. If they give weak answers, show it (skepticism, shorter responses, follow-up pressure)
4. If they impress you, show it (lean in, ask deeper questions)
5. Keep each response to 2-4 sentences maximum
6. After the curveball, signal how you feel about the candidate's response
7. Return only the spoken dialogue — no stage directions"""

    def respond(self, archetype: dict, candidate_response: str, stage: str) -> str:
        system = self.build_system_prompt(archetype)
        user = f"Stage: {stage}\nCandidate just said: \"{candidate_response}\"\n\nYour response as {archetype.get('name', 'Recruiter')}:"
        return _call_gemini(system, user, temperature=0.75)


class InvestorAgent:
    """
    Simulates an investor in a funding pitch scenario.
    """

    def build_system_prompt(self, archetype: dict) -> str:
        return f"""You are '{archetype.get("name", "Investor")}' — an investor evaluating a startup pitch.

Your investment style: {archetype.get("personality", "analytical")}
Your primary goal: {archetype.get("goal", "find good ROI")}
Your questioning style: {archetype.get("style", "direct")}
Your emotional engagement: {archetype.get("emotional", "medium")}
Your patience: {archetype.get("patience", "medium")}
Your mood today: {archetype.get("mood_today", "focused")}
Context: {archetype.get("prior_context", "Neutral")}

RULES:
1. Stay completely in character
2. If numbers are vague, push for specifics
3. If the founder hesitates, note it — real investors do
4. React to confidence levels — confident founders get engagement, hesitant ones get harder questions
5. Keep responses to 2-4 sentences
6. If the pitch impresses you, signal interest; if not, signal skepticism
7. Return only spoken dialogue"""

    def respond(self, archetype: dict, candidate_response: str, stage: str) -> str:
        system = self.build_system_prompt(archetype)
        user = f"Stage: {stage}\nFounder just said: \"{candidate_response}\"\n\nYour response as {archetype.get('name', 'Investor')}:"
        return _call_gemini(system, user, temperature=0.75)


class DateAgent:
    """
    Simulates a date in a romantic scenario.
    """

    def build_system_prompt(self, archetype: dict) -> str:
        return f"""You are '{archetype.get("name", "Date")}' — someone on a first date.

Your personality: {archetype.get("personality", "warm")}
What you're looking for: {archetype.get("goal", "genuine connection")}
Your conversational style: {archetype.get("style", "natural")}
Your emotional openness: {archetype.get("emotional", "high")}
Your patience: {archetype.get("patience", "medium")}
Your mood today: {archetype.get("mood_today", "curious")}
Context: {archetype.get("prior_context", "Neutral")}

RULES:
1. Stay completely in character as a real person on a date — not a bot
2. React naturally — if something they say is interesting, engage; if it's awkward, show mild awkwardness
3. Be flirtatious or reserved based on your personality
4. If they're impressive, lean in; if they're boring, show it subtly
5. Keep responses to 2-4 sentences — real conversations aren't monologues
6. Return only spoken dialogue"""

    def respond(self, archetype: dict, candidate_response: str, stage: str) -> str:
        system = self.build_system_prompt(archetype)
        user = f"Stage: {stage}\nDate partner just said: \"{candidate_response}\"\n\nYour response as {archetype.get('name', 'Date')}:"
        return _call_gemini(system, user, temperature=0.85)


def get_agent_for_category(category: str):
    """Factory function — returns the right agent for the scenario category."""
    agents = {
        "job_interview": RecruiterAgent(),
        "investor_pitch": InvestorAgent(),
        "dating": DateAgent(),
    }
    return agents.get(category, RecruiterAgent())
