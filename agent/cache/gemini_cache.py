"""
Gemini Context Cache Manager
=============================
Implements explicit Gemini context caching as specified in architecture.md.

Caches three things:
  1. System Instructions  — the "Coach" behavioral rules
  2. Digital Twin Persona — full JSON of user's behavior/posture/voice
  3. Success Criteria     — logic used to evaluate each simulation

Step A: Create the cache (once, after video analysis + form)
Step B: Execute 10 "light" calls, each referencing the cached context

Also integrates Redis to store the cache name for fast retrieval.
"""

import datetime
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


TWIN_COACH_SYSTEM_INSTRUCTION = """\
You are a Digital Twin Simulation Engine. You embody the exact personality,
communication style, and behavioral patterns of the user described in the
provided persona profile.

During simulations:
1. Respond EXACTLY as this person would — authentically, with their flaws and strengths
2. Never be "perfectly polished" — replicate their actual tendencies (hesitations, fillers, style)
3. Keep responses conversational (2-4 sentences) matching their natural speech patterns
4. Show the realistic stress responses, confidence levels, and communication habits described

You are NOT coaching them. You ARE them.
"""

SUCCESS_CRITERIA = {
    "job_interview": {
        "success_threshold": 7,
        "key_indicators": [
            "Clear articulation of value proposition",
            "Confident body language signals",
            "Structured STAR-method answers",
            "Appropriate assertiveness without aggression",
            "Genuine curiosity about the role",
        ],
        "failure_indicators": [
            "Vague or rambling answers",
            "Defensive under curveball questions",
            "Low confidence signals",
            "Poor storytelling",
            "No follow-up questions",
        ],
    },
    "investor_pitch": {
        "success_threshold": 7,
        "key_indicators": [
            "Clear problem-solution articulation",
            "Specific numbers and metrics",
            "Confident risk discussion",
            "Compelling vision with credible execution",
            "Strong responses to hard questions",
        ],
        "failure_indicators": [
            "Vague financials",
            "Defensive about weaknesses",
            "Overly rehearsed/unnatural",
            "Unable to handle investor pressure",
            "Weak closing ask",
        ],
    },
    "dating": {
        "success_threshold": 7,
        "key_indicators": [
            "Genuine curiosity about the other person",
            "Natural humor and warmth",
            "Appropriate vulnerability",
            "Active listening signals",
            "Confident but not arrogant self-presentation",
        ],
        "failure_indicators": [
            "Oversharing too early",
            "Emotional mismatch",
            "Too passive or too aggressive",
            "Boring or one-sided conversation",
            "Awkward silences without recovery",
        ],
    },
}


class GeminiContextCache:
    """
    Manages Gemini explicit context caching for the simulation run.
    
    Usage:
        cache_manager = GeminiContextCache()
        cache_name = cache_manager.create_cache(video_analysis, persona, form_data)
        
        # Then for each of 10 simulations:
        response = cache_manager.generate_with_cache(cache_name, scenario_prompt)
    """

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY", "")
        self.model_name = os.getenv("LLM_MODEL", "gemini-2.5-pro")
        self.available = GENAI_AVAILABLE and bool(self.api_key)
        self.client = None

        if self.available:
            self.client = genai.Client(api_key=self.api_key)

    def create_cache(
        self,
        video_analysis: dict,
        persona: dict,
        form_data: dict,
        ttl_minutes: int = 15,
    ) -> Optional[str]:
        """
        Step A: Create the Gemini context cache.
        
        Returns the cache name (used in all subsequent simulation calls).
        Falls back to None if caching is unavailable.
        """
        if not self.available:
            print("[GeminiCache] Gemini unavailable — skipping cache creation")
            return None

        # Build rich context to cache
        cached_content = self._build_cache_content(video_analysis, persona, form_data)

        try:
            cache = self.client.caches.create(
                model=self.model_name,
                config={
                    "system_instruction": TWIN_COACH_SYSTEM_INSTRUCTION,
                    "contents": [cached_content],
                    "ttl": f"{ttl_minutes * 60}s",
                },
            )
            print(f"[GeminiCache] Cache created: {cache.name} (TTL={ttl_minutes}min)")
            return cache.name

        except Exception as e:
            print(f"[GeminiCache] Cache creation error: {e} — will run without cache")
            return None

    def generate_with_cache(self, cache_name: str, scenario_prompt: str) -> str:
        """
        Step B: Run a "light" simulation call using the cached context.
        Only sends the unique scenario data — the twin persona is already cached.
        """
        if not self.available or not cache_name:
            return self._fallback_response(scenario_prompt)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=scenario_prompt,
                config={"cached_content": cache_name, "temperature": 0.7},
            )
            return response.text.strip()

        except Exception as e:
            print(f"[GeminiCache] Generation error: {e}")
            return self._fallback_response(scenario_prompt)

    def delete_cache(self, cache_name: str):
        """Clean up cache after simulation run completes."""
        if not self.available or not cache_name:
            return
        try:
            self.client.caches.delete(name=cache_name)
            print(f"[GeminiCache] Cache deleted: {cache_name}")
        except Exception as e:
            print(f"[GeminiCache] Cache deletion error: {e}")

    def _build_cache_content(
        self,
        video_analysis: dict,
        persona: dict,
        form_data: dict,
    ) -> str:
        """Build the content block to be cached — twin blueprint."""
        sections = [
            "=== DIGITAL TWIN BLUEPRINT ===",
            "",
            "--- VIDEO ANALYSIS RESULTS ---",
            json.dumps(video_analysis, indent=2, default=str)[:3000],  # trim for token limit
            "",
            "--- PERSONA PROFILE ---",
            json.dumps(persona, indent=2, default=str)[:3000],
            "",
            "--- FORM RESPONSES ---",
            json.dumps(form_data, indent=2, default=str)[:2000],
            "",
            "--- SUCCESS CRITERIA ---",
            json.dumps(SUCCESS_CRITERIA, indent=2),
            "",
            "=== END BLUEPRINT ===",
        ]
        return "\n".join(sections)

    def _fallback_response(self, scenario_prompt: str) -> str:
        return f"[Cache unavailable — scenario: {scenario_prompt[:100]}]"
