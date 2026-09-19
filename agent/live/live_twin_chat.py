"""
LiveTwinChatAgent
==================
Drives the turn-by-turn conversation once the persona of "the person the
user will face" has been built. Optionally receives a rolling snapshot of
the user's live behavioral signals (tone, posture, energy) so the persona
can react in-character to how the user is coming across — the same way a
real person subconsciously reacts to nervousness or confidence.
"""

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


class LiveTwinChatAgent:
    """Generates the next in-character reply for the live conversation."""

    def respond(
        self,
        persona: dict,
        conversation: list,
        user_message: str,
        behavior_hint: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        client = _get_client()
        # Same model used for question generation / persona building, for consistent behavior.
        model_name = os.getenv("LLM_MODEL", "gemini-2.5-pro")
        if client is None:
            return "[Live twin unavailable — Gemini not configured]"

        system = persona.get("system_prompt", "") or (
            f"You are {persona.get('persona_name','the person')}, {persona.get('persona_role','')}. "
            f"{persona.get('personality_summary','')}"
        )
        if behavior_hint:
            system += (
                f"\n\nLive read on how the user currently comes across (subtly let this color your "
                f"reaction, don't mention it explicitly): {behavior_hint}"
            )

        history_lines = []
        for m in (conversation or [])[-14:]:
            speaker = "You" if m["role"] == "twin" else "Them"
            history_lines.append(f"{speaker}: {m['content']}")
        if history_lines:
            system += "\n\nConversation so far:\n" + "\n".join(history_lines)

        try:
            from agent.token_logger import extract_token_counts, log_call
            response = client.models.generate_content(
                model=model_name,
                contents=user_message,
                config={
                    "system_instruction": system,
                    "temperature": 0.8,
                    "max_output_tokens": 1024,
                    # Minimal thinking budget — gemini-2.5-pro rejects 0, and unbounded
                    # thinking was silently eating the output budget, cutting replies off mid-sentence.
                    "thinking_config": {"thinking_budget": 128},
                },
            )
            input_tokens, output_tokens = extract_token_counts(response)
            log_call(model_name, "LiveTwinChatAgent", input_tokens, output_tokens, user_id=user_id)
            text = response.text
            if text is None:
                try:
                    text = response.candidates[0].content.parts[0].text
                except Exception:
                    text = ""
            return (text or "").strip() or "..."
        except Exception as e:
            return f"[Live twin error: {e}]"
