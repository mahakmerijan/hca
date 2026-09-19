"""
Live Digital Twin — Use Case Library
======================================
The user picks one of these right after logging in, BEFORE "Describe Them".
Each use case layers a domain-specific context on top of the existing
dynamic prompt pipeline (question generation -> persona building -> live
chat -> performance judging). The dynamic pipeline itself is unchanged —
we only append focused, domain-specific instructions to it.

TWIN_LEVELS implements the Vyakti Streak difficulty-scaling system
(Novice -> Neutral -> Hostile/Mastery), also layered on top of persona
building without touching the underlying dynamic persona logic.
"""

USE_CASES = [
    {
        "id": "dating_woman",
        "label": "Dating a Woman",
        "emoji": "💕",
        "category": "personal",
        "description": "Going on a date with women.",
        "context_prompt": (
            "This is a ROMANTIC DATING context. The twin represents a woman the user is dating or "
            "about to date. Focus on natural chemistry, playful banter, genuine curiosity about her, "
            "emotional attunement, and reading interest vs. disinterest cues. Avoid corporate or "
            "transactional language — this should feel like a real, warm human connection."
        ),
        "judge_focus": (
            "Evaluate warmth, humor, genuine curiosity about her (vs. talking only about himself), "
            "emotional attunement, and whether the user came across as confident vs. try-hard or nervous. "
            "Note any awkward silences, over-interviewing (rapid-fire questions with no reciprocation), "
            "or missed emotional cues."
        ),
        "badge_level3": "Charm Master",
    },
    {
        "id": "job_interviews",
        "label": "Job Interviews",
        "emoji": "💼",
        "category": "professional",
        "description": "Interview for a position in a job.",
        "context_prompt": (
            "This is a JOB INTERVIEW context. The twin represents a hiring manager or interviewer. Focus "
            "on professional credibility, structured answers (e.g. STAR method), concrete examples, and "
            "confidence without arrogance."
        ),
        "judge_focus": (
            "Evaluate structure and specificity of answers, evidence of preparation, professional tone, "
            "and whether the user handled behavioral/technical questions convincingly. Flag vague answers, "
            "rambling, or defensiveness under follow-up pressure."
        ),
        "badge_level3": "Interview Ace",
    },
    {
        "id": "girlfriend_conversation",
        "label": "Conversation with your Girlfriend",
        "emoji": "❤️",
        "category": "personal",
        "description": "Conversation with girlfriend on any topic.",
        "context_prompt": (
            "This is an intimate RELATIONSHIP conversation with the user's girlfriend. Focus on emotional "
            "validation, active listening, not being defensive, and resolving or discussing the topic with "
            "empathy rather than trying to 'win' the conversation."
        ),
        "judge_focus": (
            "Evaluate emotional attunement, active listening (paraphrasing, validating feelings), "
            "defensiveness vs. openness, and whether the user de-escalated or escalated tension."
        ),
        "badge_level3": "Relationship Communicator",
    },
    {
        "id": "parents_conversation",
        "label": "Conversation with your Parents",
        "emoji": "👨‍👩‍👦",
        "category": "personal",
        "description": "Conversation with your parents regarding any topic.",
        "context_prompt": (
            "This is a conversation with the user's PARENT(S). Focus on respect, patience, and balancing "
            "honesty with tact given family/generational dynamics."
        ),
        "judge_focus": (
            "Evaluate respectfulness and patience, tone with parents, and the user's ability to state their "
            "own view clearly without escalating conflict or being dismissive."
        ),
        "badge_level3": "Family Diplomat",
    },
    {
        "id": "regulatory_compliance",
        "label": "Regulatory Compliance & Auditing Interviews",
        "emoji": "📋",
        "category": "professional",
        "description": (
            "Finance and legal teams rehearse complex compliance defence drills against a simulated "
            "regulatory examiner twin to evaluate vocal tone and confidence under audit conditions."
        ),
        "context_prompt": (
            "This is a REGULATORY COMPLIANCE / AUDIT interview. The twin represents a regulatory examiner "
            "or auditor. Focus on precise, defensible answers, calm and controlled tone, no evasiveness, "
            "and command of details, documentation, and policy."
        ),
        "judge_focus": (
            "Evaluate precision and defensibility of answers, vocal composure under audit-style "
            "questioning, avoidance of evasive language, and confidence when citing evidence or policy."
        ),
        "badge_level3": "Audit Proof",
    },
    {
        "id": "medical_consultations",
        "label": "Medical Consultations & Difficult Patient Conversations",
        "emoji": "🩺",
        "category": "professional",
        "description": (
            "Doctors and healthcare administrators practice breaking bad news or managing distressed "
            "patient/family member personalities in simulated live interactions."
        ),
        "context_prompt": (
            "This is a MEDICAL CONSULTATION / difficult patient conversation. The twin represents a "
            "distressed patient or family member. Focus on empathetic delivery of difficult information, "
            "active listening, and calm reassurance without being dismissive of their fear or anger."
        ),
        "judge_focus": (
            "Evaluate empathy language, pacing when delivering difficult news, whether the user "
            "acknowledged emotions before facts, and overall bedside manner under distress."
        ),
        "badge_level3": "Compassionate Communicator",
    },
    {
        "id": "vc_investor_pitching",
        "label": "Venture Capital & Investor Pitching",
        "emoji": "📈",
        "category": "professional",
        "description": (
            "Founders simulate specific Tier-1 VC partners (e.g. analytical vs. thesis-driven "
            "personalities) to stress-test their pitches, handle objections, and refine their body language."
        ),
        "context_prompt": (
            "This is a VENTURE CAPITAL PITCH. The twin represents a Tier-1 VC partner (analytical or "
            "thesis-driven, per the user's description). Focus on crisp storytelling, data-backed claims, "
            "sharp objection handling, and composure under skeptical questioning."
        ),
        "judge_focus": (
            "Evaluate clarity of the pitch narrative, specificity of numbers/metrics, objection handling, "
            "and confidence/body language while being challenged on assumptions."
        ),
        "badge_level3": "Pitch Perfect",
    },
    {
        "id": "crisis_management_pr",
        "label": "Crisis Management & PR Handling",
        "emoji": "🎙️",
        "category": "professional",
        "description": (
            "Spokespersons face a simulated, hostile investigative journalist twin to practice managing "
            "tough, rapid-fire inquiries under intense pressure without breaking composure."
        ),
        "context_prompt": (
            "This is a CRISIS MANAGEMENT / PR interview. The twin represents a hostile investigative "
            "journalist. Focus on message discipline, staying on key messages, not getting baited into "
            "damaging admissions, and composure under rapid-fire hostile questioning."
        ),
        "judge_focus": (
            "Evaluate message discipline (bridging back to key points), composure under aggressive "
            "questioning, avoidance of defensive or evasive language, and overall command of the narrative."
        ),
        "badge_level3": "Crisis Proof",
    },
    {
        "id": "diplomatic_negotiation",
        "label": "High-Level Diplomatic & Cross-Cultural Negotiation",
        "emoji": "🌐",
        "category": "professional",
        "description": (
            "Professionals prepare for international high-stakes deals by practicing with a twin "
            "configured to mirror specific cultural communication styles, decision-making pacing, and "
            "friction points."
        ),
        "context_prompt": (
            "This is a HIGH-LEVEL DIPLOMATIC / CROSS-CULTURAL NEGOTIATION. The twin represents a "
            "counterpart from the specific cultural/negotiation style described by the user. Focus on "
            "cultural sensitivity, patience with different decision-making pacing, and tactfully "
            "identifying and navigating friction points."
        ),
        "judge_focus": (
            "Evaluate cultural sensitivity, patience with pacing/formality differences, the user's ability "
            "to navigate friction points diplomatically, and whether they adapted their style appropriately."
        ),
        "badge_level3": "Master Diplomat",
    },
    {
        "id": "corporate_leadership_boardrooms",
        "label": "High-Stakes Corporate Leadership & Boardrooms",
        "emoji": "🏛️",
        "category": "professional",
        "description": (
            "Executives simulate difficult Q&A sessions with a digital twin of an aggressive investor or "
            "skeptical board member before live earnings calls or funding pitches."
        ),
        "context_prompt": (
            "This is a HIGH-STAKES CORPORATE BOARDROOM / earnings-call Q&A. The twin represents an "
            "aggressive investor or skeptical board member. Focus on command of numbers, composure under "
            "tough Q&A, and executive presence."
        ),
        "judge_focus": (
            "Evaluate command of financial/strategic detail, composure under adversarial questioning, "
            "executive presence (posture, voice authority), and whether answers were direct vs. evasive."
        ),
        "badge_level3": "Boardroom Ready",
    },
]

_BY_ID = {uc["id"]: uc for uc in USE_CASES}


def get_use_case(use_case_id: str) -> dict | None:
    return _BY_ID.get(use_case_id)


def list_use_cases() -> list:
    return USE_CASES


# ── Vyakti Streak difficulty scaling (Novice -> Neutral -> Hostile/Mastery) ──
TWIN_LEVELS = {
    1: {
        "name": "Novice Twin",
        "behavior": (
            "DIFFICULTY: NOVICE. Be cooperative and encouraging. Give the user gentle hints if they "
            "stumble, speak slowly and clearly, avoid interrupting, and stay patient and warm even if "
            "their answer is weak."
        ),
        "pressure_level": 3,
        "openness_level": 8,
    },
    2: {
        "name": "Neutral Twin",
        "behavior": (
            "DIFFICULTY: NEUTRAL. Maintain a standard, professional demeanor. Neutral pushback, normal "
            "follow-up questions, and react naturally to strong or weak answers — don't go out of your "
            "way to be difficult or overly kind."
        ),
        "pressure_level": 5,
        "openness_level": 5,
    },
    3: {
        "name": "Hostile/Mastery Twin",
        "behavior": (
            "DIFFICULTY: HOSTILE/MASTERY. Be demanding and challenging. Interrupt the user if they ramble, "
            "bring up tough or trick details, act impatient with vague answers, push back hard, and test "
            "their composure under real pressure — without becoming abusive or unrealistic."
        ),
        "pressure_level": 9,
        "openness_level": 3,
    },
}


def get_twin_level(level: int) -> dict:
    try:
        level = int(level)
    except (TypeError, ValueError):
        level = 1
    return TWIN_LEVELS.get(level, TWIN_LEVELS[1])
