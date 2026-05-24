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
