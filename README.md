# 🧠 Human Behavior Analysis AI Agent

An AI-powered agent that analyzes a person's behavior from video and predicts their probability of success in three key scenarios:

- 💼 **Cracking a Job Interview**
- 🤝 **Getting a Business Deal**
- ❤️ **Going on a Date**

## How It Works

The agent processes a video file through **four analysis pipelines**:

### 1. 😊 Facial Expression Analysis
- Detects emotions frame-by-frame using **DeepFace**
- Tracks emotion distribution (happy, neutral, sad, angry, etc.)
- Calculates smile ratio and emotional consistency

### 2. 🧍 Body Language Analysis
- Uses **MediaPipe Pose** to track body posture
- Evaluates shoulder alignment, head position, spine straightness
- Detects confidence signals (open/closed posture, arm crossing, lean direction)
- Analyzes hand gestures using **MediaPipe Hands**

### 3. 🎤 Voice & Speech Analysis
- Extracts audio from video using **MoviePy**
- Analyzes pitch, energy, and speaking pace with **Librosa**
- Detects pauses and speech rhythm
- Transcribes speech and analyzes content (filler words, vocabulary richness, confident language)

### 4. 🔮 Prediction Engine
- Combines all analysis scores with scenario-specific weights
- Generates probability scores (0-100%) for each scenario
- Builds a behavioral profile with strengths and areas for improvement

## Installation

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Place your video file as `video.mp4` in the project root (or specify a path)

2. Run the analysis:
```bash
python main.py
```

3. Options:
```bash
python main.py --video path/to/your/video.mp4   # Custom video path
python main.py --sample-rate 15                   # Analyze more frames (slower but more accurate)
python main.py --config custom_config.json        # Custom configuration
```

## Output

The agent produces:
- **Terminal output**: Beautiful formatted tables with predictions, score breakdowns, and behavioral profile
- **Text report**: `output/analysis_report.txt`
- **JSON data**: `output/analysis_data.json` (full raw analysis data)

### Sample Output
```
🎯 COMPATIBILITY PREDICTIONS
┌────────────────────────┬──────────────┬────────┬──────────────────────────────┐
│ Scenario               │ Probability  │ Grade  │ Verdict                      │
├────────────────────────┼──────────────┼────────┼──────────────────────────────┤
│ 💼 Cracking a Job      │    72.3%     │  B+    │ Strong candidate!            │
│ 🤝 Business Deal       │    68.5%     │  B     │ Decent chance, room to grow  │
│ ❤️  Going on a Date    │    81.2%     │  A     │ Strong candidate!            │
└────────────────────────┴──────────────┴────────┴──────────────────────────────┘
```

## Question research suggestions

In the web UI, every preparation and follow-up question has a **Suggestions**
link, with no separate recommended-source buttons. It opens Google in a new tab using the selected
use case, your **Who are you about to face?** description, and that specific
question. Editing the description updates the follow-up research links without
changing your answers. Older scenario questions use their scenario description.
Profile questionnaires use their profile section instead, without the scenario.

The same search includes relevant website domains as source hints. These are
combined with `OR` and a broader use-case alternative, rather than `site:`
restrictions, so other websites remain eligible. Google controls the results:
the hints cannot guarantee that any particular recommended website appears.

| Use case | Recommended sources |
|----------|---------------------|
| Job interviews | AmbitionBox, Glassdoor, Indeed |
| Dating | Bumble, Hinge, The Gottman Institute |
| Girlfriend conversations | The Gottman Institute, Psychology Today, Relate |
| Parent conversations | Psychology Today, Greater Good, AAMFT |
| Regulatory compliance | SEC, FCA, The IIA |
| Medical consultations | AMA, AAFP, NICE |
| Investor pitching | Y Combinator, Sequoia Capital, Andreessen Horowitz |
| Crisis management and PR | PRSA, CIPR, Institute for Public Relations |
| Diplomatic negotiation | Harvard Program on Negotiation, Cultural Atlas, UNITAR |
| Corporate leadership | Harvard Business Review, NACD, Institute of Directors |

New use cases retain unrestricted Suggestions searches; add their recommended
domains to `getQuestionResearchSites` in `templates/index.html`.

The UI warns that Google receives the description, use case, and question when
you open a research link. Avoid confidential or sensitive details. Follow-up
answers are not appended. No Google request is made merely by generating links.
Google receives the website hints as well. Google controls availability and ranking; links do not guarantee answers,
replace professional advice, or establish private facts about someone.

Run the dependency-free UI regression checks with:

```bash
node --test tests/question_suggestions.test.cjs
```

## Twin gender and live-call voice

The first follow-up question in the Live Digital Twin flow is always the compulsory
**"Is this person male or female?"** (Male / Female chips). It is injected server-side in
`agent/live/live_question_generator.py`, and any gender question the LLM generates is removed.
The twin cannot be built until it is answered. The answer is stored as `persona.gender`, and
the live call speaks with a matching browser voice (inferred from well-known voice names). If
no matching voice is installed, pitch is adjusted instead.

## First-visit guided tutorials

After login, each screen offers a short highlighted walkthrough with **Next**, **Back**,
and **Skip tutorial**. Tutorials cover the welcome screens, use-case selection, describing
the person, follow-up answers and Suggestions, building a twin, hands-free calls, and the
judgment report. Follow-up questions and completed reports get separate tours when those
sections become available; seeing an empty report page does not count as seeing the report tour.

- Completed or skipped sections are remembered in this browser's local storage, shared across
  accounts using that browser. Clearing site data makes the first-visit guides appear again.
- **Show tutorial**, at the bottom-left, replays the available guidance for the current screen.
- Welcome timers and the introduction video pause while their tutorial is open.
- Tours never start during an active call, and replay is disabled until the call ends.
- Escape skips the current tutorial; keyboard focus stays inside the guide and returns on close.
- If browser storage is blocked, the guide shows a notice and remembers completion only until
  the page reloads. No account, description, answer, or transcript is stored by the tutorial.

The controller and step definitions live in `static/guided-tour.js`; presentation is in
`static/guided-tour.css`. Navigation and dynamic content rendering in `templates/index.html`
notify the controller directly, without polling or adding an external tour dependency.

## Live-call voice input

Live calls are hands-free. Click **Start Live Call**, hear the twin's opening line, then speak
when the status says **Listening**. There are no Send or microphone buttons in the live chat.
The read-only transcript grows as you speak, and your answer is submitted automatically after
**1.5 seconds of silence**. Transcription and reply generation add network/processing delay
after that pause; the twin's voice does not necessarily start within 1.5 seconds.

Calls use turn-taking: speech-to-text is paused while the twin speaks or generates its reply,
then resumes automatically after playback ends. Speech during the twin's turn is not submitted.
This avoids recording the twin's own voice as a user reply. Ending a call or navigating away
cancels listening, playback and pending replies; late results cannot restart the conversation.
Voice/transcription failures are shown below the transcript.

Live calls use **server-backed transcription** from the start, avoiding the initial browser
speech-service failure in VS Code's/Electron's built-in browser:

- While you speak, the phrase in progress is re-transcribed every ~0.35 s of new audio (one
  request in flight at a time) via `POST /live/transcribe` (`agent/live/live_transcriber.py`,
  using the `SpeechRecognition` library). The words in the box grow as you talk, about once a second.
- After 1.5 s of silence, capture stops and the last phrase is transcribed before submitting
  the complete answer once. Long answers are processed in segments of up to roughly 12 s,
  preferring a quiet point in the last 1.5 s; these segment boundaries do not submit the answer.
- If a phrase's final transcription comes back empty or fails, the words already shown for it are
  kept, so earlier speech is never erased.
- The live-call transcript is multi-line and grows with the text up to a scroll cap. Submitted
  answers also remain visible in the conversation.

Follow-up question answers still have manual microphone buttons. They use the browser's
built-in speech recognition where available, with automatic server fallback when the speech
service is unavailable. Their server capture uses a 0.7 s phrase pause and continues until
the user clicks stop; question answers are not submitted automatically.

Word-by-word streaming with no network round-trip needs either a browser with a speech service
(desktop Chrome or Edge) or a streaming speech-to-text API (e.g. Google Cloud Speech-to-Text
streaming over WebSockets). That API isn't set up in this project.

## Configuration

Edit `config.json` to customize:
- **frame_sample_rate**: How many frames to skip between analyses (higher = faster, lower = more accurate)
- **weights**: How much each behavioral dimension contributes to each scenario
- **output settings**: Where to save reports

## Project Structure

```
human_compatibility_analysis/
├── main.py                          # Entry point
├── config.json                      # Configuration
├── requirements.txt                 # Python dependencies
├── video.mp4                        # Input video
├── agent/
│   ├── __init__.py
│   ├── behavior_agent.py            # Main AI agent orchestrator
│   ├── video_processor.py           # Video frame & audio extraction
│   └── analyzers/
│       ├── __init__.py
│       ├── facial_expression.py     # Emotion detection (DeepFace)
│       ├── body_language.py         # Posture & gesture analysis (MediaPipe)
│       └── voice_speech.py          # Voice tone & speech analysis (Librosa)
└── output/
    ├── analysis_report.txt          # Human-readable report
    ├── analysis_data.json           # Full JSON data
    └── audio.wav                    # Extracted audio
```

## Tech Stack

| Component | Library |
|-----------|---------|
| Face/Emotion Detection | DeepFace, OpenCV |
| Pose Estimation | MediaPipe |
| Audio Processing | Librosa, MoviePy |
| Speech-to-Text | SpeechRecognition (Google API) |
| UI/Output | Rich (terminal formatting) |

## Notes

- The analysis quality depends on video quality (good lighting, clear face visibility)
- Speech transcription requires an internet connection (uses Google Speech API)
- For best results, use videos where the person is facing the camera and speaking
- Lower `frame_sample_rate` (e.g., 10-15) gives more accurate results but takes longer
