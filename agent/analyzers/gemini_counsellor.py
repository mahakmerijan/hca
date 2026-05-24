"""
Gemini AI Counsellor Module — v2
Uses Google Gemini to act as an infinite-scale communication coach.

Analyses across three layers:
  1. Verbal Intelligence   (clarity, structure, tone, persuasion)
  2. Non-Verbal Intelligence (micro-expressions, facial cues, eye movement)
  3. Behavioral Signals    (gestures, body language, confidence markers)

Scenarios:
  • Closing a Deal
  • Getting Promoted
  • Delivering a Presentation
  • Going on a Date
"""

import json
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


COACH_SYSTEM_PROMPT = """\
You are a world-class communication coach and behavioural psychologist who scales
infinitely. You analyse humans across THREE layers:

  1. VERBAL INTELLIGENCE    — clarity, structure, tone, persuasion, vocabulary
  2. NON-VERBAL INTELLIGENCE — micro-expressions, facial cues, eye movement patterns
  3. BEHAVIORAL SIGNALS     — gestures, body language, confidence markers

You review computer-vision + audio analysis data for a person's video and produce
a highly personalised, actionable coaching report.

You may also receive user-supplied context:
  • A job description   → tailor advice for "Getting Promoted"
  • A business idea     → tailor advice for "Closing a Deal"
  • A date partner desc → tailor advice for "Going on a Date"

RULES:
1. Write in second person ("you", "your").
2. Be empathetic but brutally honest — like the best coach money can buy.
3. Reference SPECIFIC data (e.g. "your smile ratio was only 9%", "arms crossed 40% of the time").
4. For every scenario give a 0-100 probability AND rich, personalised coaching advice.
5. Structure the 3-layer analysis with real findings — do NOT make things up.
6. Return ONLY valid JSON — no markdown fences, no preamble, no trailing text.
7. For moment_analysis: use the EMOTION TIMELINE to identify specific timestamps
   where the detected emotion is misaligned with the context (e.g. fear/sadness
   during a pitch, anger during a date). Cross-reference with transcript phrases
   if available. For each moment give the exact timestamp, what was off, the
   transcript fragment if relevant, and rewrite how that sentence should be
   delivered with the correct expression/tone for each applicable scenario.

Return EXACTLY this JSON structure:
{
  "overall_summary": "3-4 sentence holistic summary.",
  "personality_snapshot": "2-3 sentences on personality.",

  "three_layer_analysis": {
    "verbal_intelligence": {
      "score": <0-100>,
      "summary": "2-3 sentences on verbal performance.",
      "what_worked": ["point 1", "point 2"],
      "what_didnt": ["point 1", "point 2"],
      "improvements": ["actionable improvement 1", "actionable improvement 2", "actionable improvement 3"]
    },
    "non_verbal_intelligence": {
      "score": <0-100>,
      "summary": "2-3 sentences.",
      "what_worked": ["point 1", "point 2"],
      "what_didnt": ["point 1", "point 2"],
      "improvements": ["improvement 1", "improvement 2", "improvement 3"]
    },
    "behavioral_signals": {
      "score": <0-100>,
      "summary": "2-3 sentences.",
      "what_worked": ["point 1", "point 2"],
      "what_didnt": ["point 1", "point 2"],
      "improvements": ["improvement 1", "improvement 2", "improvement 3"]
    }
  },

  "strengths": [
    { "title": "Short title", "detail": "2-3 sentence explanation." }
  ],
  "weaknesses": [
    { "title": "Short title", "detail": "2-3 sentence explanation." }
  ],

  "scenarios": {
    "closing_deal": {
      "probability": <0-100>,
      "verdict": "One-line verdict",
      "what_worked": "What specific signals help close deals.",
      "what_didnt": "What is holding you back.",
      "coaching_advice": "4-6 sentences of actionable deal-closing coaching."
    },
    "getting_promoted": {
      "probability": <0-100>,
      "verdict": "One-line verdict",
      "what_worked": "What signals project leadership potential.",
      "what_didnt": "What undermines your promotion candidacy.",
      "coaching_advice": "4-6 sentences of actionable promotion coaching."
    },
    "delivering_presentation": {
      "probability": <0-100>,
      "verdict": "One-line verdict",
      "what_worked": "What makes your presentation style compelling.",
      "what_didnt": "What detracts from impact.",
      "coaching_advice": "4-6 sentences of actionable presentation coaching."
    },
    "going_on_date": {
      "probability": <0-100>,
      "verdict": "One-line verdict",
      "what_worked": "What makes you engaging on a date.",
      "what_didnt": "What might make you seem less appealing.",
      "coaching_advice": "4-6 sentences of actionable dating coaching."
    }
  },

  "improvement_plan": [
    {
      "area": "Area name",
      "priority": "high | medium | low",
      "current_state": "What the data shows now.",
      "target_state": "What excellent looks like.",
      "action_steps": ["step 1", "step 2", "step 3"],
      "training_loop": "A repeatable daily or weekly drill to build this skill."
    }
  ],

  "moment_analysis": [
    {
      "timestamp": "0:12",
      "emotion_detected": "fear",
      "what_was_said": "Exact or approximate phrase from transcript at this moment, or describe what was happening visually.",
      "why_its_off": "Why this expression/emotion is misaligned — e.g. fear signals during a confident pitch undermine credibility.",
      "better_delivery": "How exactly to say/deliver this moment differently — tone, facial expression, posture, pace, phrasing.",
      "scenario_relevance": ["closing_deal", "getting_promoted"]
    }
  ],

  "coach_verdict": "The single most important thing this person must change — one direct sentence.",
  "closing_message": "Warm, encouraging 2-3 sentence close."
}

For moment_analysis: produce 4-8 of the most impactful moments from the timeline.
Pick moments where the emotion is clearly misaligned (fear, sadness, anger, disgust
during professional/dating contexts). If transcript is available, quote the exact
phrases. For each moment, write a vivid redelivery description — not just what
to feel but HOW to physically express it (jaw relaxed, eyes wide, slight smile,
slowed pace, lower pitch, etc). scenario_relevance should list which scenarios
this moment most impacts.
"""


def _build_analysis_context(results: dict, user_context: dict = None) -> str:
    lines = []

    vi = results.get("video_info", {})
    fps = float(vi.get("fps", 25) or 25)
    frame_sample_rate = int(vi.get("frame_sample_rate", 1) or 1)
    lines.append(f"VIDEO: {vi.get('duration_seconds','?')}s, {vi.get('resolution','?')}, {fps:.1f} fps")

    fa = results.get("facial_analysis", {})
    lines.append("\nFACIAL EXPRESSION ANALYSIS (Non-Verbal Layer):")
    lines.append(f"  Frames analysed: {fa.get('total_frames_analyzed', 0)}")
    lines.append(f"  Faces detected: {fa.get('faces_detected', 0)}")
    lines.append(f"  Smile ratio: {fa.get('smile_ratio', 0):.1%}")
    emo = fa.get("emotion_distribution", {})
    if emo:
        lines.append(f"  Emotion distribution: {json.dumps({k: round(v*100,1) for k,v in emo.items()})}%")
    scores = fa.get("scenario_scores", {})
    if scores:
        lines.append(f"  Scenario scores: job={scores.get('job_interview',0)}, biz={scores.get('business_deal',0)}, date={scores.get('date',0)}")

    # Timestamped emotion timeline — convert frame_idx to MM:SS
    timeline = fa.get("emotion_timeline", [])
    if timeline:
        lines.append("  Emotion timeline (timestamp → emotion):")
        for entry in timeline[:40]:  # up to 40 keyframes
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                frame_idx, emotion = entry
                secs = int(frame_idx * frame_sample_rate / fps)
                mm, ss = divmod(secs, 60)
                lines.append(f"    {mm}:{ss:02d} → {emotion}")

    bl = results.get("body_language_analysis", {})
    lines.append("\nBODY LANGUAGE ANALYSIS (Behavioral Signals Layer):")
    lines.append(f"  Pose detected in {bl.get('pose_detected_count',0)} / {bl.get('total_frames_analyzed',0)} frames")
    am = bl.get("average_metrics", {})
    if am:
        lines.append(f"  Shoulder alignment: {am.get('avg_shoulder_alignment',0):.2f}")
        lines.append(f"  Head uprightness: {am.get('avg_head_uprightness',0):.2f}")
        lines.append(f"  Posture openness: {am.get('avg_openness',0):.2f}")
        lines.append(f"  Confidence score: {am.get('avg_confidence_score',0):.2f}")
        lines.append(f"  Arms crossed ratio: {am.get('arms_crossed_ratio',0):.2f}")
    bscores = bl.get("scenario_scores", {})
    if bscores:
        lines.append(f"  Scenario scores: job={bscores.get('job_interview',0)}, biz={bscores.get('business_deal',0)}, date={bscores.get('date',0)}")

    vs = results.get("voice_speech_analysis", {})
    vf = vs.get("audio_features", {})
    lines.append("\nVOICE & SPEECH ANALYSIS (Verbal Intelligence Layer):")
    if vf.get("audio_loaded"):
        sr = vf.get("speaking_rate", {})
        lines.append(f"  Speaking pace: {sr.get('pace_label','?')} ({sr.get('estimated_syllables_per_sec',0):.1f} syll/s)")
        pi = vf.get("pitch", {})
        lines.append(f"  Mean pitch: {pi.get('mean_pitch',0):.0f} Hz, stability: {pi.get('pitch_stability',0):.2f}")
        pa = vf.get("pauses", {})
        lines.append(f"  Pauses: {pa.get('num_pauses',0)}, ratio: {pa.get('pause_ratio',0):.2f}")
        sc = vf.get("speech_content", {})
        lines.append(f"  Words: {sc.get('word_count',0)}, filler ratio: {sc.get('filler_word_ratio',0):.2%}")
        lines.append(f"  Vocabulary richness: {sc.get('vocabulary_richness',0):.2f}")
        lines.append(f"  Confident language ratio: {sc.get('confident_language_ratio',0):.2f}")
        # Pass full transcript so Gemini can quote exact phrases in moment_analysis
        full_transcript = vs.get("full_transcript", "") or sc.get("transcript_preview", "")
        if full_transcript:
            lines.append(f"  Full transcript: \"{full_transcript}\"")
    else:
        lines.append("  No audio available.")

    pred = results.get("final_predictions", {})
    lines.append("\nCOMPUTED PREDICTIONS (weighted model):")
    for s in ["job_interview", "business_deal", "date"]:
        d = pred.get(s, {})
        lines.append(f"  {s}: {d.get('probability',0):.1f}% (grade {d.get('grade','?')})")

    if user_context:
        jd = user_context.get("job_description", "")
        bi = user_context.get("business_idea", "")
        dp = user_context.get("date_partner", "")
        if jd or bi or dp:
            lines.append("\nUSER-SUPPLIED CONTEXT:")
            if jd:
                lines.append(f"\nJOB DESCRIPTION:\n{jd}")
            if bi:
                lines.append(f"\nBUSINESS IDEA / DEAL CONTEXT:\n{bi}")
            if dp:
                lines.append(f"\nDATE PARTNER DESCRIPTION:\n{dp}")

    return "\n".join(lines)


class GeminiCounsellor:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY", "")
        self.model_name = os.getenv("LLM_MODEL", "gemini-2.5-pro")
        self.available = False

        if not GENAI_AVAILABLE:
            print("[GeminiCounsellor] google-genai not installed.")
            return
        if not self.api_key:
            print("[GeminiCounsellor] GOOGLE_API_KEY not set in .env")
            return

        self.client = genai.Client(api_key=self.api_key)
        self.available = True
        print(f"[GeminiCounsellor] Ready — model={self.model_name}")

    def generate_counselling(self, analysis_results: dict, user_context: dict = None) -> dict:
        if not self.available:
            return self._fallback(analysis_results)

        context = _build_analysis_context(analysis_results, user_context)
        user_prompt = (
            "Here is the complete behavioural analysis for a person's video. "
            "Produce your full coaching report as specified.\n\n"
            f"{context}"
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config={
                    "system_instruction": COACH_SYSTEM_PROMPT,
                    "temperature": 0.7,
                },
            )
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3].strip()
            if text.startswith("json"):
                text = text[4:].strip()

            report = json.loads(text)
            return report

        except json.JSONDecodeError as e:
            print(f"[GeminiCounsellor] JSON parse error: {e}")
            return self._fallback(analysis_results)
        except Exception as e:
            print(f"[GeminiCounsellor] Gemini call failed: {e}")
            return self._fallback(analysis_results)

    def _fallback(self, results: dict) -> dict:
        pred = results.get("final_predictions", {})
        prof = results.get("behavioral_profile", {})
        bp = lambda k: pred.get(k, {}).get("probability", 50)
        avg = int((bp("job_interview") + bp("business_deal")) / 2)

        return {
            "overall_summary": prof.get("overall_impression", "Analysis complete."),
            "personality_snapshot": f"Dominant emotion: {prof.get('dominant_emotion','neutral')}.",
            "three_layer_analysis": {
                "verbal_intelligence": {"score": 50, "summary": "Verbal data being processed.", "what_worked": [], "what_didnt": [], "improvements": []},
                "non_verbal_intelligence": {"score": 50, "summary": "Non-verbal data being processed.", "what_worked": [], "what_didnt": [], "improvements": []},
                "behavioral_signals": {"score": 50, "summary": "Behavioral data being processed.", "what_worked": [], "what_didnt": [], "improvements": []}
            },
            "strengths": [{"title": s, "detail": s} for s in prof.get("strengths", [])] or [{"title": "Baseline competency", "detail": "Shows baseline competency."}],
            "weaknesses": [{"title": w, "detail": w} for w in prof.get("areas_for_improvement", [])] or [{"title": "No major issues", "detail": "No major concerns identified."}],
            "scenarios": {
                "closing_deal": {"probability": bp("business_deal"), "verdict": "Moderate chance", "what_worked": "Shows engagement.", "what_didnt": "Needs stronger persuasion signals.", "coaching_advice": "Focus on structured pitching and active listening to close deals more effectively."},
                "getting_promoted": {"probability": bp("job_interview"), "verdict": "Moderate chance", "what_worked": "Projects some authority.", "what_didnt": "Needs stronger executive presence.", "coaching_advice": "Work on projecting confidence, clarity, and strategic thinking."},
                "delivering_presentation": {"probability": avg, "verdict": "Moderate chance", "what_worked": "Has foundational presence.", "what_didnt": "Needs vocal variety.", "coaching_advice": "Focus on vocal variety, pausing for effect, and strong eye contact."},
                "going_on_date": {"probability": bp("date"), "verdict": "Moderate chance", "what_worked": "Appears approachable.", "what_didnt": "Needs more warmth.", "coaching_advice": "Smile more authentically and be present in conversations."}
            },
            "improvement_plan": [],
            "coach_verdict": "Focus on projecting warmth and confidence simultaneously.",
            "closing_message": "Every interaction is a chance to grow. Keep practising!"
        }
