"""
Persona Generator
=================
Uses Gemini 2.5 Pro to create a structured LLM-based Digital Twin persona
from the built profile. Also handles explicit Gemini context caching.
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


PERSONA_SYSTEM_PROMPT = """\
You are an expert behavioural psychologist and AI persona architect.
Your job is to create a precise, simulation-ready Digital Twin persona from the
provided profile data. This persona will be used to simulate hundreds of
real-world social interactions (job interviews, investor pitches, dating).

Return ONLY valid JSON in this exact structure:
{
  "twin_name": "Digital Twin of [Name]",
  "persona_summary": "3-4 sentence description of this person.",
  "system_prompt": "The full system prompt to use when running this twin as an LLM agent. Write it in second person and include ALL key personality nuances.",
  "personality_dimensions": {
    "openness": <0-100>,
    "conscientiousness": <0-100>,
    "extraversion": <0-100>,
    "agreeableness": <0-100>,
    "neuroticism": <0-100>
  },
  "communication_fingerprint": {
    "response_length": "short | medium | long",
    "directness": "direct | diplomatic | passive",
    "emotional_expressiveness": "high | medium | low",
    "use_of_humor": "frequent | occasional | rare",
    "technical_vocabulary": "high | medium | low",
    "filler_word_tendency": "high | medium | low",
    "interruption_tendency": "high | medium | low"
  },
  "strengths_in_simulation": ["strength 1", "strength 2", "strength 3"],
  "weaknesses_in_simulation": ["weakness 1", "weakness 2", "weakness 3"],
  "scenario_specific": {
    "job_interview": {
      "likely_behaviors": ["behavior 1", "behavior 2"],
      "likely_failure_points": ["failure 1", "failure 2"],
      "confidence_level": <0-100>
    },
    "investor_pitch": {
      "likely_behaviors": ["behavior 1", "behavior 2"],
      "likely_failure_points": ["failure 1", "failure 2"],
      "confidence_level": <0-100>
    },
    "dating": {
      "likely_behaviors": ["behavior 1", "behavior 2"],
      "likely_failure_points": ["failure 1", "failure 2"],
      "confidence_level": <0-100>
    }
  },
  "embodied_signals": {
    "posture_profile": "description of typical posture",
    "eye_contact_pattern": "description",
    "gesture_style": "description",
    "voice_characteristics": "description"
  }
}
"""


class PersonaGenerator:
    """Generates a Digital Twin persona using Gemini 2.5 Pro."""

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY", "")
        self.model_name = os.getenv("LLM_MODEL", "gemini-2.5-pro")
        self.available = GENAI_AVAILABLE and bool(self.api_key)

        if self.available:
            self.client = genai.Client(api_key=self.api_key)

    def generate(self, profile: dict) -> dict:
        """Generate a full persona from a built twin profile."""
        if not self.available:
            return self._fallback(profile)

        # Build a structured, human-readable prompt that highlights every data source
        bm = profile.get("behavioral_model", {})
        cm = profile.get("cognitive_model", {})
        em = profile.get("embodied_model", {})
        sr = em.get("self_reported", {})
        vd = em.get("video_derived", {})
        ai = em.get("ai_coach_assessment", "")
        km = em.get("key_expression_moments", [])
        sp = em.get("success_probabilities", {})
        name = profile.get("name", "User")

        def _s(v): return str(v) if v else "not provided"
        def _list(v): return ", ".join(v) if isinstance(v, list) and v else "not provided"

        narrative_section = f"""
=== SELF-ASSESSMENT NARRATIVE (their own words) ===
Biggest strength: {_s(bm.get('biggest_strength'))}
Biggest weakness: {_s(bm.get('biggest_weakness'))}
How others see them: {_s(bm.get('how_others_see_you'))}
Past failure they learned from: {_s(bm.get('past_failure'))}
Career story: {_s(cm.get('career_story'))}
How they pitch themselves: {_s(cm.get('pitch_yourself'))}
What blocks them: {_s(cm.get('what_blocks_you'))}
Ideal relationship dynamic: {_s(cm.get('ideal_relationship'))}
Nervous tells (body language when anxious): {_s(sr.get('nervous_tells'))}
Confident tells (body language when at ease): {_s(sr.get('confident_tells'))}
How they think they come across on first impression: {_s(sr.get('first_impression'))}
"""

        behavioral_section = f"""
=== BEHAVIORAL MODEL (questionnaire) ===
Personality traits: {_list(bm.get('personality_traits'))}
Introvert/Extrovert score: {bm.get('introvert_extrovert_score', 5)} / 10
Risk-taking score: {bm.get('risk_taking_score', 5)} / 10
Communication style: {_s(bm.get('communication_style'))}
Decision style: {_s(bm.get('decision_style'))}
Social energy: {_s(bm.get('social_energy'))}
Conflict response: {_s(bm.get('conflict_response'))}
Daily habits: {_list(bm.get('daily_habits'))}
Morning routine: {_s(bm.get('morning_routine'))}
"""

        cognitive_section = f"""
=== COGNITIVE / DECISION MODEL (questionnaire) ===
Career goal: {_s(cm.get('career_goal'))}
Core values: {_list(cm.get('core_values'))}
Long-term vision: {_s(cm.get('long_term_vision'))}
Investment style: {_s(cm.get('investment_style'))}
Business risk tolerance: {_s(cm.get('business_risk_tolerance'))}
Stress response: {_s(cm.get('stress_response'))}
Learning style: {_s(cm.get('learning_style'))}
Ideal workplace: {_list(cm.get('ideal_workplace'))}
Negotiation style: {_s(cm.get('negotiation_style'))}
Dating preferences: {_list(cm.get('dating_preferences'))}
"""

        video_section = ""
        if vd:
            video_section = f"""
=== EMBODIED MODEL (extracted from video analysis) ===
Posture openness: {_s(vd.get('posture_openness'))}
Spine/head uprightness: {_s(vd.get('spine_angle'))}
Shoulder alignment: {_s(vd.get('shoulder_alignment'))}
Arms crossed ratio: {_s(vd.get('arms_crossed_ratio'))}
Overall confidence score (video): {_s(vd.get('confidence_score'))}
Smile ratio: {_s(vd.get('smile_ratio'))}
Dominant emotion: {_s(vd.get('dominant_emotion'))}
Emotion distribution: {json.dumps(vd.get('emotion_distribution', {}))}
Speaking pace: {_s(vd.get('speaking_pace'))}
Mean pitch (Hz): {_s(vd.get('mean_pitch_hz'))}
Pitch stability: {_s(vd.get('pitch_stability'))}
Filler word ratio: {_s(vd.get('filler_word_ratio'))}
Vocabulary richness: {_s(vd.get('vocabulary_richness'))}
Confident language ratio: {_s(vd.get('confident_language_ratio'))}
Word count: {_s(vd.get('word_count'))}
Full transcript (what they actually said): {_s(vd.get('full_transcript'))}
"""

        if km:
            video_section += "\nKey expression moments flagged by AI coach:\n"
            for m in km:
                video_section += f"  - [{m.get('timestamp','?')}] {m.get('emotion','')} — {m.get('what_was_said','')}\n    Issue: {m.get('issue','')}\n    Better delivery: {m.get('suggestion','')}\n"

        if sp:
            video_section += f"\nSuccess probabilities (from video): {json.dumps(sp)}\n"

        # Inject the richest coaching data directly
        coach_verdict = em.get("coach_verdict", "")
        coach_strengths = em.get("coach_strengths", [])
        coach_weaknesses = em.get("coach_weaknesses", [])
        improvement_plan = em.get("improvement_plan", [])
        personality_snapshot = em.get("personality_snapshot", "")

        if personality_snapshot:
            video_section += f"\nAI personality snapshot: {personality_snapshot}\n"
        if coach_verdict:
            video_section += f"\nCoach's single most important observation: {coach_verdict}\n"
        if coach_strengths:
            video_section += "\nVideo-observed strengths:\n"
            for s in coach_strengths[:5]:
                video_section += f"  - {s.get('title','')}: {s.get('detail','')}\n"
        if coach_weaknesses:
            video_section += "\nVideo-observed weaknesses:\n"
            for w in coach_weaknesses[:5]:
                video_section += f"  - {w.get('title','')}: {w.get('detail','')}\n"
        if improvement_plan:
            video_section += "\nAI improvement plan (from video):\n"
            for p in improvement_plan[:4]:
                video_section += f"  - [{p.get('priority','?')} priority] {p.get('area','')}: {p.get('current_state','')} → Target: {p.get('target_state','')}\n"

        if ai:
            # Only include the 3-layer analysis summary (not the full JSON to keep token count down)
            three_layer = ai.get("three_layer_analysis", {}) if isinstance(ai, dict) else {}
            if three_layer:
                video_section += "\n=== THREE-LAYER AI ANALYSIS ===\n"
                for layer, data in three_layer.items():
                    if isinstance(data, dict):
                        video_section += f"  {layer} (score {data.get('score','?')}/100): {data.get('summary','')}\n"
                        if data.get("what_didnt"):
                            video_section += f"    Weaknesses: {'; '.join(data['what_didnt'][:2])}\n"

        user_prompt = (
            f"Create a simulation-ready Digital Twin persona for: {name}\n"
            f"{narrative_section}"
            f"{behavioral_section}"
            f"{cognitive_section}"
            f"{video_section}"
            "\nIMPORTANT: Use ALL the above data — especially the narrative self-assessment "
            "and video-derived signals — to make the system_prompt and persona as accurate "
            "and authentic as possible. The twin must sound, behave, and respond exactly like "
            "this specific person."
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config={
                    "system_instruction": PERSONA_SYSTEM_PROMPT,
                    "temperature": 0.6,
                },
            )
            text = response.text.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3].strip()
            if text.startswith("json"):
                text = text[4:].strip()

            persona = json.loads(text)
            persona["source_profile"] = profile
            return persona

        except Exception as e:
            print(f"[PersonaGenerator] Error: {e}")
            return self._fallback(profile)

    def _fallback(self, profile: dict) -> dict:
        """Return a basic persona when Gemini is unavailable."""
        bm = profile.get("behavioral_model", {})
        cm = profile.get("cognitive_model", {})
        ie_score = bm.get("introvert_extrovert_score", 5)
        extraversion = int(ie_score * 10)

        name = profile.get("name", "User")
        traits = bm.get("personality_traits", [])
        comm = bm.get("communication_style", "Diplomatic")

        system_prompt = (
            f"You are a digital twin of {name}.\n"
            f"Traits: {', '.join(traits) if traits else 'Balanced personality'}\n"
            f"Communication style: {comm}\n"
            f"Risk tolerance: {bm.get('risk_taking_score', 5)}/10\n"
            f"Core values: {', '.join(cm.get('core_values', []))}\n"
            f"Career goal: {cm.get('career_goal', 'Professional growth')}\n"
            "Respond authentically as this person would in real conversations."
        )

        return {
            "twin_name": f"Digital Twin of {name}",
            "persona_summary": f"A {'introverted' if ie_score < 5 else 'extroverted'} individual with traits: {', '.join(traits[:3])}.",
            "system_prompt": system_prompt,
            "personality_dimensions": {
                "openness": 60,
                "conscientiousness": 65,
                "extraversion": extraversion,
                "agreeableness": 60,
                "neuroticism": 40,
            },
            "communication_fingerprint": {
                "response_length": "medium",
                "directness": "diplomatic",
                "emotional_expressiveness": "medium",
                "use_of_humor": "occasional",
                "technical_vocabulary": "medium",
                "filler_word_tendency": "medium",
                "interruption_tendency": "low",
            },
            "strengths_in_simulation": ["Thoughtful responses", "Self-awareness", "Consistency"],
            "weaknesses_in_simulation": ["May hesitate under pressure", "Could be more assertive"],
            "scenario_specific": {
                "job_interview": {"likely_behaviors": ["Prepared"], "likely_failure_points": ["Nervousness"], "confidence_level": 60},
                "investor_pitch": {"likely_behaviors": ["Logical"], "likely_failure_points": ["Risk aversion framing"], "confidence_level": 55},
                "dating": {"likely_behaviors": ["Attentive"], "likely_failure_points": ["Overthinking"], "confidence_level": 60},
            },
            "embodied_signals": {
                "posture_profile": "Moderately open posture",
                "eye_contact_pattern": "Moderate eye contact",
                "gesture_style": "Occasional hand gestures",
                "voice_characteristics": "Measured, moderate pace",
            },
            "source_profile": profile,
        }
