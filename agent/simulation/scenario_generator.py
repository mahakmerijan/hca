"""
Scenario Generator
==================
Generates 10 unique counter-party scenarios as specified in architecture.md:
  - 3 Job Interview variations
  - 3 Investor Pitch variations
  - 4 Dating variations

Each scenario has a unique counter-party persona with different personality and goals.
"""

import random
from typing import List, Dict

# ── Job Interview archetypes (300 variations) ─────────────────────

JOB_ARCHETYPES = [
    {"name": "The Hardball Manager", "personality": "aggressive", "goal": "find weakness under pressure",
     "style": "blunt", "emotional": "low", "patience": "low"},
    {"name": "The Friendly HR", "personality": "warm", "goal": "culture fit assessment",
     "style": "conversational", "emotional": "high", "patience": "high"},
    {"name": "The Bored Executive", "personality": "disengaged", "goal": "rubber stamp process",
     "style": "distracted", "emotional": "low", "patience": "medium"},
    {"name": "The Technical Skeptic", "personality": "analytical", "goal": "verify competence deeply",
     "style": "precise", "emotional": "low", "patience": "medium"},
    {"name": "The Empathetic Leader", "personality": "supportive", "goal": "assess growth mindset",
     "style": "mentoring", "emotional": "high", "patience": "high"},
    {"name": "The Speed Interviewer", "personality": "efficient", "goal": "rapid screening",
     "style": "rapid-fire", "emotional": "medium", "patience": "low"},
    {"name": "The Culture Champion", "personality": "values-driven", "goal": "alignment check",
     "style": "story-based", "emotional": "high", "patience": "high"},
    {"name": "The Numbers Obsessor", "personality": "data-driven", "goal": "quantify everything",
     "style": "metric-focused", "emotional": "low", "patience": "medium"},
    {"name": "The Passive Questioner", "personality": "shy", "goal": "fill mandatory slots",
     "style": "quiet", "emotional": "medium", "patience": "high"},
    {"name": "The Devil's Advocate", "personality": "contrarian", "goal": "test resilience",
     "style": "challenging", "emotional": "medium", "patience": "low"},
]

INVESTOR_ARCHETYPES = [
    {"name": "The Numbers-Only VC", "personality": "analytical", "goal": "ROI maximization",
     "style": "financial", "emotional": "low", "patience": "low"},
    {"name": "The Visionary Angel", "personality": "idealistic", "goal": "big-picture impact",
     "style": "inspirational", "emotional": "high", "patience": "medium"},
    {"name": "The Skeptical Partner", "personality": "cautious", "goal": "risk assessment",
     "style": "probing", "emotional": "low", "patience": "medium"},
    {"name": "The Serial Entrepreneur", "personality": "experienced", "goal": "operational validation",
     "style": "practical", "emotional": "medium", "patience": "low"},
    {"name": "The Impact Investor", "personality": "mission-driven", "goal": "social value",
     "style": "holistic", "emotional": "high", "patience": "high"},
    {"name": "The Tech Enthusiast VC", "personality": "tech-forward", "goal": "innovation potential",
     "style": "technical", "emotional": "medium", "patience": "medium"},
    {"name": "The Bored Partner", "personality": "saturated", "goal": "quick decision",
     "style": "dismissive", "emotional": "low", "patience": "very low"},
    {"name": "The Mentor-Investor", "personality": "nurturing", "goal": "founder potential",
     "style": "coaching", "emotional": "high", "patience": "high"},
    {"name": "The Dealmaker", "personality": "transactional", "goal": "terms negotiation",
     "style": "direct", "emotional": "low", "patience": "low"},
    {"name": "The Community VC", "personality": "network-focused", "goal": "ecosystem fit",
     "style": "relationship", "emotional": "medium", "patience": "medium"},
]

DATING_ARCHETYPES = [
    {"name": "The Introverted Match", "personality": "shy", "goal": "genuine connection",
     "style": "quiet", "emotional": "high", "patience": "high"},
    {"name": "The High-Energy Match", "personality": "extroverted", "goal": "fun and excitement",
     "style": "enthusiastic", "emotional": "high", "patience": "medium"},
    {"name": "The Intellectual", "personality": "cerebral", "goal": "mental stimulation",
     "style": "philosophical", "emotional": "medium", "patience": "high"},
    {"name": "The Practical Partner", "personality": "grounded", "goal": "long-term compatibility",
     "style": "direct", "emotional": "medium", "patience": "medium"},
    {"name": "The Romantic", "personality": "emotional", "goal": "deep emotional connection",
     "style": "poetic", "emotional": "very high", "patience": "medium"},
    {"name": "The Career-Focused", "personality": "ambitious", "goal": "compatible life goals",
     "style": "structured", "emotional": "low", "patience": "low"},
    {"name": "The Free Spirit", "personality": "spontaneous", "goal": "adventure and fun",
     "style": "unpredictable", "emotional": "high", "patience": "low"},
    {"name": "The Cautious Dater", "personality": "guarded", "goal": "trust-building",
     "style": "reserved", "emotional": "medium", "patience": "high"},
    {"name": "The Social Butterfly", "personality": "charismatic", "goal": "chemistry check",
     "style": "playful", "emotional": "high", "patience": "medium"},
    {"name": "The Values-Aligner", "personality": "principled", "goal": "shared worldview",
     "style": "probing", "emotional": "medium", "patience": "high"},
]

CURVEBALL_QUESTIONS = {
    "job": [
        "What's your biggest professional failure and what did you learn?",
        "Why should I hire you over someone with more experience?",
        "Tell me about a time you disagreed with your manager — what did you do?",
        "Where do you see yourself in 5 years — and be honest, not polished.",
        "What's something you're terrible at, professionally?",
        "Walk me through a time you missed a deadline. What happened?",
        "You have competing priorities and limited resources — how do you decide?",
        "Give me an example where you had to influence without authority.",
    ],
    "investor": [
        "What happens if a bigger player copies your idea in 6 months?",
        "Why you? What makes you the right person to execute this?",
        "Walk me through your worst-case financial scenario.",
        "Who's tried this before and why did they fail?",
        "What's your customer acquisition cost and lifetime value?",
        "How do you handle the risk of key person dependency?",
        "What's your exit strategy?",
        "Why are you raising this amount and not more or less?",
    ],
    "dating": [
        "What does your relationship with your family look like?",
        "What ended your last relationship?",
        "What do you want out of life, really?",
        "How do you handle conflict in relationships?",
        "Are you actually ready for something serious right now?",
        "What are your deal-breakers?",
        "How important is physical intimacy in a relationship to you?",
        "What does your ideal Saturday look like?",
    ],
}


class ScenarioGenerator:
    """
    Generates 10 unique micro-simulation scenarios.
    Distribution: 3 job interviews, 3 investor pitches, 4 dating.
    """

    def generate_all(self) -> List[Dict]:
        """Generate the complete set of 10 scenarios."""
        scenarios = []

        # 3 job interview scenarios
        for i in range(3):
            archetype = JOB_ARCHETYPES[i % len(JOB_ARCHETYPES)]
            variation = self._add_variation(archetype, i)
            scenarios.append(self._build_scenario(
                scenario_id=i + 1,
                category="job_interview",
                archetype=variation,
                curveball=random.choice(CURVEBALL_QUESTIONS["job"]),
            ))

        # 3 investor pitch scenarios
        for i in range(3):
            archetype = INVESTOR_ARCHETYPES[i % len(INVESTOR_ARCHETYPES)]
            variation = self._add_variation(archetype, i)
            scenarios.append(self._build_scenario(
                scenario_id=4 + i,
                category="investor_pitch",
                archetype=variation,
                curveball=random.choice(CURVEBALL_QUESTIONS["investor"]),
            ))

        # 4 dating scenarios
        for i in range(4):
            archetype = DATING_ARCHETYPES[i % len(DATING_ARCHETYPES)]
            variation = self._add_variation(archetype, i)
            scenarios.append(self._build_scenario(
                scenario_id=7 + i,
                category="dating",
                archetype=variation,
                curveball=random.choice(CURVEBALL_QUESTIONS["dating"]),
            ))

        return scenarios

    def generate_batch(self, category: str, count: int) -> List[Dict]:
        """Generate a specific batch of scenarios for a category."""
        archetypes = {
            "job_interview": JOB_ARCHETYPES,
            "investor_pitch": INVESTOR_ARCHETYPES,
            "dating": DATING_ARCHETYPES,
        }.get(category, JOB_ARCHETYPES)

        curveballs = CURVEBALL_QUESTIONS.get(
            {"job_interview": "job", "investor_pitch": "investor", "dating": "dating"}.get(category, "job")
        )

        return [
            self._build_scenario(
                scenario_id=i + 1,
                category=category,
                archetype=self._add_variation(archetypes[i % len(archetypes)], i),
                curveball=random.choice(curveballs),
            )
            for i in range(count)
        ]

    def _add_variation(self, archetype: dict, seed: int) -> dict:
        """Add small variations to an archetype to create unique instances."""
        rng = random.Random(seed)
        variation = dict(archetype)
        variation["mood_today"] = rng.choice(["focused", "tired", "energized", "distracted", "impatient", "curious"])
        variation["time_pressure"] = rng.choice(["high", "medium", "low"])
        variation["prior_context"] = rng.choice([
            "Has seen 10 candidates today",
            "Just had a bad meeting",
            "Running 15 minutes behind",
            "Very interested from your resume",
            "Skeptical based on your background",
            "Neutral — clean slate",
        ])
        variation["instance_id"] = seed
        return variation

    def _build_scenario(self, scenario_id: int, category: str, archetype: dict, curveball: str) -> dict:
        return {
            "scenario_id": scenario_id,
            "category": category,
            "counter_party": archetype,
            "opening_prompt": self._opening_for(category, archetype),
            "curveball_question": curveball,
            "closing_prompt": self._closing_for(category, archetype),
            "success_criteria": self._success_criteria(category, archetype),
        }

    def _opening_for(self, category: str, archetype: dict) -> str:
        name = archetype.get("name", "Interviewer")
        if category == "job_interview":
            return f"Tell me about yourself and why you're interested in this role."
        elif category == "investor_pitch":
            return f"You have 3 minutes. Pitch me your idea."
        else:
            return f"So, what made you swipe right? Tell me a bit about yourself."

    def _closing_for(self, category: str, archetype: dict) -> str:
        if category == "job_interview":
            return "Do you have any questions for me? Why should we pick you over other candidates?"
        elif category == "investor_pitch":
            return "What's your ask and what will you do with the money in the first 90 days?"
        else:
            return "I had a good time. Would you want to do this again?"

    def _success_criteria(self, category: str, archetype: dict) -> dict:
        base = {"alignment": 0.4, "friction": 0.3, "outcome": 0.3}
        # Adjust weights by archetype personality
        if archetype.get("emotional") == "high":
            base["alignment"] = 0.5
            base["outcome"] = 0.3
        elif archetype.get("emotional") == "low":
            base["alignment"] = 0.25
            base["friction"] = 0.4
            base["outcome"] = 0.35
        return base
