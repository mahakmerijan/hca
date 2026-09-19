"""
Digital Twin Questionnaire Form Schema
=======================================
Covers sections a, b, c from the architecture:
  a. Behavioral Model (How you act)
  b. Cognitive / Decision Model (How you think)
  c. Embodied Model (How you present yourself) — derived from video
"""

TWIN_FORM_SCHEMA = {
    "sections": [
        {
            "id": "behavioral",
            "title": "Behavioral Model",
            "subtitle": "How you act",
            "icon": "🧬",
            "questions": [
                {
                    "id": "introvert_extrovert",
                    "label": "Where do you fall on the introvert-extrovert spectrum?",
                    "type": "scale",
                    "min": 1,
                    "max": 10,
                    "min_label": "Deeply Introverted",
                    "max_label": "Highly Extroverted",
                    "required": True
                },
                {
                    "id": "risk_taking",
                    "label": "How would you rate your risk-taking tendency?",
                    "type": "scale",
                    "min": 1,
                    "max": 10,
                    "min_label": "Very Risk-Averse",
                    "max_label": "Love Taking Risks",
                    "required": True
                },
                {
                    "id": "personality_traits",
                    "label": "Select all personality traits that describe you",
                    "type": "multi_select",
                    "options": [
                        "Analytical", "Creative", "Empathetic", "Decisive",
                        "Competitive", "Collaborative", "Detail-oriented",
                        "Big-picture thinker", "Optimistic", "Realistic",
                        "Confrontational", "Avoids conflict", "Leadership-driven",
                        "Follower", "Spontaneous", "Structured"
                    ],
                    "required": True
                },
                {
                    "id": "morning_routine",
                    "label": "Describe your typical morning routine in 2-3 sentences",
                    "type": "textarea",
                    "placeholder": "e.g. I wake up at 6am, exercise for 30 minutes, then spend 20 minutes reading...",
                    "required": False
                },
                {
                    "id": "communication_style",
                    "label": "How do you usually communicate?",
                    "type": "single_select",
                    "options": [
                        "Direct and blunt — I say what I mean",
                        "Diplomatic — I consider others' feelings",
                        "Analytical — I use data and logic",
                        "Storytelling — I use narratives and examples",
                        "Passive — I tend to agree to avoid conflict",
                        "Assertive — I speak confidently but respectfully"
                    ],
                    "required": True
                },
                {
                    "id": "decision_style",
                    "label": "How do you make important decisions?",
                    "type": "single_select",
                    "options": [
                        "Fast — I trust my gut",
                        "Deliberate — I analyze all options thoroughly",
                        "Collaborative — I consult others before deciding",
                        "Avoidant — I delay difficult decisions",
                        "Data-driven — I research extensively first"
                    ],
                    "required": True
                },
                {
                    "id": "social_energy",
                    "label": "After a long day of meetings/socializing, you feel:",
                    "type": "single_select",
                    "options": [
                        "Energized — I thrive on social interaction",
                        "Neutral — depends on the people",
                        "Drained — I need alone time to recharge",
                        "Exhausted — socializing is very taxing for me"
                    ],
                    "required": True
                },
                {
                    "id": "conflict_response",
                    "label": "When someone disagrees with you strongly, you typically:",
                    "type": "single_select",
                    "options": [
                        "Stand your ground firmly",
                        "Try to find a middle ground",
                        "Defer to avoid tension",
                        "Get frustrated and emotional",
                        "Ask clarifying questions first"
                    ],
                    "required": True
                },
                {
                    "id": "daily_habits",
                    "label": "Which of these are part of your regular routine? (select all)",
                    "type": "multi_select",
                    "options": [
                        "Daily exercise", "Meditation/mindfulness", "Journaling",
                        "Reading books", "Networking events", "Side projects",
                        "Social media scrolling", "Early sleeping", "Late nights",
                        "Meal prepping", "Continuous learning/courses"
                    ],
                    "required": False
                },
                {
                    "id": "biggest_strength",
                    "label": "What do you consider your single biggest personal strength in social situations? Give a real example.",
                    "type": "textarea",
                    "placeholder": "e.g. I'm very good at making people feel comfortable. In my last job interview I noticed the interviewer was tense, so I made a light joke and we immediately relaxed into a real conversation…",
                    "required": False
                },
                {
                    "id": "biggest_weakness",
                    "label": "What is your biggest weakness in social or professional interactions? Be honest — this helps your twin improve.",
                    "type": "textarea",
                    "placeholder": "e.g. I tend to over-explain and ramble when I'm nervous. In presentations I've noticed people start losing focus after my first 2 minutes…",
                    "required": False
                },
                {
                    "id": "how_others_see_you",
                    "label": "How do others typically describe you? (what have friends, colleagues, or partners said about your personality?)",
                    "type": "textarea",
                    "placeholder": "e.g. My friends say I'm very intense and passionate but can come across as intimidating. My manager told me I'm great technically but need to work on small talk…",
                    "required": False
                },
                {
                    "id": "past_failure",
                    "label": "Describe a situation where a conversation or interaction went badly. What happened and why do you think it failed?",
                    "type": "textarea",
                    "placeholder": "e.g. I froze during a pitch to investors when they asked about our burn rate. I hadn't prepared a confident answer and I stumbled, which lost their trust…",
                    "required": False
                }
            ]
        },
        {
            "id": "cognitive",
            "title": "Cognitive & Decision Model",
            "subtitle": "How you think",
            "icon": "🧠",
            "questions": [
                {
                    "id": "career_goal",
                    "label": "What is your primary career ambition in the next 3 years?",
                    "type": "single_select",
                    "options": [
                        "Get promoted to senior/management role",
                        "Switch careers / industries",
                        "Start my own business",
                        "Become an expert/specialist",
                        "Work remotely / achieve work-life balance",
                        "Secure a high-paying role"
                    ],
                    "required": True
                },
                {
                    "id": "dating_preference",
                    "label": "In a romantic partner, what do you value most?",
                    "type": "multi_select",
                    "options": [
                        "Intellectual compatibility", "Physical attraction",
                        "Emotional intelligence", "Ambition and drive",
                        "Shared values", "Humor", "Stability and reliability",
                        "Adventure and spontaneity", "Kindness and empathy"
                    ],
                    "required": True
                },
                {
                    "id": "business_risk_tolerance",
                    "label": "For a business idea you believe in, how much would you risk?",
                    "type": "single_select",
                    "options": [
                        "My entire savings if necessary",
                        "Up to 50% of my savings",
                        "Only money I can afford to lose (< 20%)",
                        "I'd only invest with external funding",
                        "I would not risk personal money"
                    ],
                    "required": True
                },
                {
                    "id": "core_values",
                    "label": "Select your top 5 core values",
                    "type": "multi_select",
                    "max_select": 5,
                    "options": [
                        "Honesty", "Loyalty", "Ambition", "Freedom", "Security",
                        "Family", "Success", "Creativity", "Justice", "Fun",
                        "Growth", "Integrity", "Power", "Compassion", "Wisdom",
                        "Adventure", "Stability", "Recognition"
                    ],
                    "required": True
                },
                {
                    "id": "long_term_goal",
                    "label": "Describe your 10-year life vision in 2-3 sentences",
                    "type": "textarea",
                    "placeholder": "Where do you see yourself in 10 years? What does success look like?",
                    "required": False
                },
                {
                    "id": "investment_style",
                    "label": "If you had $100,000 to invest, you would:",
                    "type": "single_select",
                    "options": [
                        "Put it all in safe index funds",
                        "Split 70/30 between safe and high-risk",
                        "Invest in my own business idea",
                        "Invest in a friend's startup",
                        "Keep it as savings/emergency fund",
                        "Diversify across stocks, crypto, real estate"
                    ],
                    "required": True
                },
                {
                    "id": "stress_response",
                    "label": "When under significant pressure or stress, you:",
                    "type": "single_select",
                    "options": [
                        "Perform even better — pressure brings out my best",
                        "Manage well but feel the strain",
                        "Become anxious and overthink",
                        "Withdraw and go quiet",
                        "Seek support from others immediately"
                    ],
                    "required": True
                },
                {
                    "id": "learning_style",
                    "label": "How do you learn best?",
                    "type": "single_select",
                    "options": [
                        "By doing — hands-on experience",
                        "By reading and researching",
                        "By watching/listening to others",
                        "By teaching or explaining to others",
                        "Through structured courses and programs"
                    ],
                    "required": True
                },
                {
                    "id": "ideal_workplace",
                    "label": "Your ideal work environment is:",
                    "type": "multi_select",
                    "options": [
                        "Fast-paced startup", "Established corporation",
                        "Remote-first", "Highly collaborative team",
                        "Independent/solo work", "Creative environment",
                        "Data and analytics focused", "Customer-facing",
                        "Entrepreneurial / founder-led"
                    ],
                    "required": True
                },
                {
                    "id": "negotiation_style",
                    "label": "In a salary or business negotiation, you tend to:",
                    "type": "single_select",
                    "options": [
                        "State your number boldly and hold firm",
                        "Start high and negotiate down",
                        "Wait for the other side to offer first",
                        "Feel uncomfortable and accept quickly",
                        "Prepare extensively and negotiate point-by-point"
                    ],
                    "required": True
                },
                {
                    "id": "career_story",
                    "label": "In 3-4 sentences, describe your career journey so far and where you want to go. Be specific about industries, roles, or ambitions.",
                    "type": "textarea",
                    "placeholder": "e.g. I started as a software engineer at a large bank, moved to a Series B fintech startup where I led a team of 5, and now I want to transition into a product management role at a consumer tech company…",
                    "required": False
                },
                {
                    "id": "pitch_yourself",
                    "label": "How would you pitch yourself in 60 seconds to a recruiter or investor? Write it out as you would actually say it.",
                    "type": "textarea",
                    "placeholder": "e.g. Hi, I'm Alex — I've spent 5 years solving payment infrastructure problems for fintech companies. I reduced transaction failures by 40% at my last company and grew our API adoption 3x in 18 months. I'm now looking for a company where I can do that at a larger scale…",
                    "required": False
                },
                {
                    "id": "ideal_relationship",
                    "label": "Describe your ideal relationship dynamic. What kind of partner are you and what do you need from a partner?",
                    "type": "textarea",
                    "placeholder": "e.g. I'm a very independent person who needs alone time, but I also crave deep intellectual conversations. I tend to be the planner in relationships. I need someone who respects my space but is emotionally available when I do open up…",
                    "required": False
                },
                {
                    "id": "what_blocks_you",
                    "label": "What is the one mental block, fear, or habit that most holds you back professionally or socially?",
                    "type": "textarea",
                    "placeholder": "e.g. Imposter syndrome. Even after 8 years of experience I still feel like I don't belong in senior rooms. I downplay my achievements and hesitate to claim credit…",
                    "required": False
                }
            ]
        },
        {
            "id": "embodied",
            "title": "Embodied Self-Assessment",
            "subtitle": "How you present yourself (self-report, cross-referenced with video)",
            "icon": "🫀",
            "questions": [
                {
                    "id": "self_eye_contact",
                    "label": "How often do you maintain eye contact during conversations?",
                    "type": "scale",
                    "min": 1,
                    "max": 10,
                    "min_label": "Rarely — I look away often",
                    "max_label": "Always — strong eye contact",
                    "required": True
                },
                {
                    "id": "self_posture",
                    "label": "How would you describe your typical posture?",
                    "type": "single_select",
                    "options": [
                        "Upright and open — I stand/sit tall",
                        "Slightly slouched when relaxed",
                        "Arms often crossed or closed",
                        "Leaning forward — very engaged",
                        "Variable — depends on my confidence level"
                    ],
                    "required": True
                },
                {
                    "id": "self_gestures",
                    "label": "How much do you use hand gestures when speaking?",
                    "type": "scale",
                    "min": 1,
                    "max": 10,
                    "min_label": "Very little — hands stay still",
                    "max_label": "Constantly — I talk with my hands",
                    "required": True
                },
                {
                    "id": "self_smile",
                    "label": "How often do you naturally smile in conversations?",
                    "type": "scale",
                    "min": 1,
                    "max": 10,
                    "min_label": "Rarely smile",
                    "max_label": "Smile very often",
                    "required": True
                },
                {
                    "id": "self_voice_confidence",
                    "label": "How confident does your voice sound when you present or pitch?",
                    "type": "scale",
                    "min": 1,
                    "max": 10,
                    "min_label": "Shaky/hesitant",
                    "max_label": "Commanding and clear",
                    "required": True
                },
                {
                    "id": "self_perceived_confidence",
                    "label": "Overall, how confident do others perceive you?",
                    "type": "scale",
                    "min": 1,
                    "max": 10,
                    "min_label": "Others see me as shy/reserved",
                    "max_label": "Others see me as very confident",
                    "required": True
                },
                {
                    "id": "nervous_tells",
                    "label": "When you're nervous or uncomfortable, what does your body do? (physical tells you've noticed)",
                    "type": "textarea",
                    "placeholder": "e.g. I fidget with my ring, I avoid eye contact and look at the floor, my voice gets quieter and I speak faster, I cross my arms without realising it…",
                    "required": False
                },
                {
                    "id": "confident_tells",
                    "label": "When you feel confident and 'in flow', how does your body language and voice change?",
                    "type": "textarea",
                    "placeholder": "e.g. I lean forward, make steady eye contact, my voice becomes deeper and slower, I use my hands more, and I smile naturally…",
                    "required": False
                },
                {
                    "id": "first_impression",
                    "label": "What first impression do you think you make on strangers? Has anyone ever surprised you by describing you differently than you expected?",
                    "type": "textarea",
                    "placeholder": "e.g. I think I come across as serious and reserved. But a colleague once told me that I seemed arrogant in our first meeting, which surprised me because I was just nervous…",
                    "required": False
                }
            ]
        }
    ]
}
