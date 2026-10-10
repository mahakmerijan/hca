# HCA End-to-End Flow and Architecture

This document describes what the current codebase does from the user's first screen through video coaching, Digital Twin creation, simulation, and post-simulation feedback. It follows the Flask application in `app.py` and `templates/index.html`; it does not treat every aspiration in `architecture.md` as already implemented. It also covers the `live-video` branch's real-time Live Digital Twin call feature (see "Live Digital Twin — Real-Time Call" below), which is the branch currently deployed to production.

## At a Glance

The app has three connected experiences:

1. **Video communication coach:** analyzes one or more video uploads and returns a Gemini coaching report plus rule-based scores.
2. **Digital Twin simulator:** combines that video analysis with a behavioral, cognitive, and embodied self-assessment; Gemini builds the user's persona; the simulation service runs either 10 benchmark scenarios or one custom target-persona conversation; a separate analysis step groups outcomes and generates coaching feedback.
3. **Live Digital Twin (real-time call):** the user picks a practice use case, describes a real person they're about to face, answers follow-up questions, and then has an actual live webcam/mic conversation with an AI embodying that person — graded afterward on real, live-measured behavioral signals rather than a pre-recorded video.

The video analysis is a prerequisite for first-time Digital Twin creation. The initial coach report is available before simulation; simulation feedback is a later, separate report. The Live Digital Twin call (experience 3) does not require a pre-recorded video at all — it builds its persona purely from the user's free-text description and follow-up answers.

## User Journey

```mermaid
flowchart TD
    A[Open app] --> B[Register or log in]
    B --> C{Choose experience}
    C -->|Analyze my video| D[Optional job, business, or date context]
    C -->|Talk with Digital Twin| E[Complete behavioral, cognitive, and embodied questionnaire]
    D --> F[Create video analysis job]
    E --> G[Upload one or more videos]
    G --> F
    F --> H[Background video and audio analysis]
    H --> I[Rule-based scores and Gemini coach report]
    I --> J[View three-layer report, strengths, probabilities, plan, and key moments]
    I --> K{Digital Twin path?}
    K -->|Yes| L[Merge questionnaire with completed video results]
    L --> M[Build structured profile]
    M --> N[Gemini creates persona]
    N --> O{Simulation type}
    O -->|Benchmark| P[Generate 10 predefined scenarios]
    O -->|Targeted| Q[Generate a custom counter-party persona]
    P --> R[Run background conversations and grade each scenario]
    Q --> R
    R --> S[Poll progress and show conversations]
    S --> T[Run failure clustering and coaching generation]
    T --> U[Show simulation report, successes, failures, and drills]
    U --> V[Save report/memory for later insights]
```

## System Architecture

```mermaid
flowchart LR
    subgraph Browser[Browser: templates/index.html]
        UI[Login gate, onboarding, video coach, twin questionnaire, simulation tabs, reports]
        Local[localStorage: auth token, user/twin IDs, some drafts and video job IDs]
        UI <--> Local
    end

    subgraph Flask[Flask application: app.py]
        Routes[HTTP routes and JSON responses]
        Jobs[Process-local jobs and user_contexts dictionaries]
        Worker[Daemon thread: video analysis]
        Services[Lazy service initialization]
    end

    subgraph Domain[Domain services]
        User[UserService: account, password hash, JWT]
        Analysis[AnalysisService: failure clusters, feedback, insights]
        Twin[TwinService: profile, persona, twin persistence]
        Simulation[SimulationService: scenarios, progress, results]
    end

    subgraph Video[Video analysis pipeline]
        Decode[OpenCV: video metadata and sampled frames]
        Audio[MoviePy: extract WAV]
        Face[DeepFace: frame emotion and smile metrics]
        Pose[MediaPipe Pose and Hands: posture and gesture metrics]
        Voice[Librosa and SpeechRecognition: voice/audio and transcript features]
        Scores[Weighted scenario scores and behavioral profile]
        Coach[GeminiCounsellor: narrative coaching JSON]
        Decode --> Face
        Decode --> Pose
        Audio --> Voice
        Face --> Scores
        Pose --> Scores
        Voice --> Scores
        Scores --> Coach
    end

    subgraph TwinPipeline[Digital Twin pipeline]
        Schema[Questionnaire schema]
        Builder[TwinProfileBuilder]
        Persona[PersonaGenerator: Gemini]
        TwinMemory[FeedbackMemoryManager: twin profile memory]
        Schema --> Builder
        Builder --> Persona
        Persona --> TwinMemory
    end

    subgraph SimulationPipeline[Simulation pipeline]
        ScenarioGen[ScenarioGenerator: fixed 3 job, 3 investor, 4 dating scenarios]
        TwinLLM[Digital Twin response: Gemini]
        Counter[Recruiter, Investor, or Date agent: Gemini]
        Referee[RefereeAgent: per-scenario grading]
        ScenarioGen --> TwinLLM
        TwinLLM <--> Counter
        TwinLLM --> Referee
        Counter --> Referee
    end

    subgraph Coaching[Post-simulation coaching]
        Cluster[FailureClusterAnalyzer: Deep Agents/LangGraph when available, fallbacks]
        Feedback[FeedbackGenerator: Gemini]
        Memory[Semantic, episodic, and procedural LangGraph memory]
        Cluster --> Feedback
        Cluster --> Memory
        Feedback --> Memory
    end

    subgraph Persistence[Configured persistence and fallbacks]
        PG[(PostgreSQL when POSTGRES_URI is configured)]
        Redis[(Redis cache/checkpointer when REDIS_URL is configured)]
        File[(output/twins_store.json fallback for twins)]
        RAM[(Process-local dictionaries and LangGraph in-memory stores)]
    end

    subgraph Google[Google model services]
        Gemini[Vertex AI Gemini via configured model names]
        Embeddings[Optional Google text-embedding-004]
    end

    UI --> Routes
    Routes --> Jobs
    Routes --> Services
    Jobs --> Worker
    Worker --> Video
    Routes --> User
    Routes --> Twin
    Routes --> Simulation
    Routes --> Analysis
    Coach --> Routes
    Coach --> Builder
    Twin --> TwinPipeline
    Simulation --> SimulationPipeline
    Analysis --> Coaching
    Persona --> Gemini
    TwinLLM --> Gemini
    Counter --> Gemini
    Referee --> Gemini
    Coach --> Gemini
    Feedback --> Gemini
    Memory --> Embeddings
    User --> PG
    Twin --> PG
    Twin --> File
    Simulation --> PG
    Analysis --> PG
    Services --> Redis
    Memory --> Redis
    Memory --> PG
    Services --> RAM
    Memory --> RAM
```

## End-to-End Request Flow

### 1. Login and mode selection

The UI presents registration/login and then the onboarding choice. `/auth/register` and `/auth/login` call `UserService`; passwords are PBKDF2-HMAC hashed and successful login returns a JWT (or a development fallback token if PyJWT is unavailable). The browser keeps the token and user/twin IDs in `localStorage`. Twin and simulation routes require a bearer token; the video job routes themselves do not currently apply `require_auth`.

### 2. Video upload and analysis

Both the standalone video-coach experience and the Digital Twin video panel use the same analysis worker.

1. The browser calls `POST /init-job` to allocate a job ID.
2. In the coach experience, it optionally sends text context to `/save-context` and PDF/DOC/TXT context files to `/upload-context-file`.
3. It sends the video to `POST /upload`. The route accepts MP4, AVI, MOV, MKV, WEBM, FLV, and WMV up to 500 MB, saves the upload to a temporary file, and starts `_run_analysis` in a daemon thread.
4. The browser polls `GET /status/<job_id>`. When the job is done, it requests `GET /results/<job_id>`.
5. `_run_analysis` extracts metadata and sampled frames using OpenCV, extracts audio with MoviePy, then analyzes facial expressions, posture/hand gestures, and voice/speech. It computes weighted scores and a basic behavioral profile, then calls `GeminiCounsellor` for the narrative report.
6. The worker returns the result in the process-local `jobs` dictionary and deletes the temporary video and extracted WAV in its `finally` block.

The web upload path samples every 90th frame by default (about every three seconds at 30 FPS). The CLI path in `main.py` is separate and uses the configuration file/default settings.

### 3. Video analysis and initial coach report

The deterministic analyzers create measurable inputs; the counselor turns those inputs into a narrative report.

| Input path | Current implementation | Main outputs |
|---|---|---|
| Face | DeepFace emotion analysis over sampled frames | Emotion distribution/timeline, smile ratio, scenario scores |
| Body | MediaPipe Pose and Hand Landmarkers | Head/spine proxy, shoulder alignment, openness, crossed-arm/confidence proxies, hand visibility and gesture activity |
| Voice | Librosa plus SpeechRecognition | Pace, pitch, pauses, transcript and transcript-derived indicators when available |
| Score | Weighted calculation in `BehaviorAnalysisAgent` | Job interview, business deal, and date probabilities with score breakdowns |
| Coaching | Gemini counselor prompt | Three communication layers, strengths/weaknesses, scenario probabilities, improvement plan, key moments, and top priority |

The coach UI renders the report and can export it to PDF. The Gemini report is a model-generated interpretation of extracted signals; the probabilities are assessments, not validated real-world outcome guarantees.

### 4. Questionnaire and Digital Twin creation

The UI loads the public schema from `GET /twin/schema`. It contains three sections:

- **Behavioral model:** traits, introversion/extroversion, risk tendency, habits, communication/decision style, conflict response, self-described strengths and weaknesses.
- **Cognitive/decision model:** career ambition, dating preferences, risk tolerance, values, investment and negotiation style, stress response, work preferences, goals, and narrative answers.
- **Embodied self-assessment:** self-rated eye contact, posture, gestures, smile, voice confidence, plus nervous/confident tells and first-impression narrative.

When the user submits, the UI sends `POST /twin/create` with at least eight non-empty answers and one or more completed video job IDs. The route merges available analysis results, then `TwinProfileBuilder` creates `behavioral_model`, `cognitive_model`, and `embodied_model`. The video-derived model includes posture/gesture metrics, emotion/timeline, voice metrics, transcript, counselor assessment, and key moments where those values exist.

`PersonaGenerator` sends the structured profile to Gemini using `LLM_MODEL` (default `gemini-2.5-pro`). Its JSON result includes a persona summary, system prompt, personality dimensions, communication fingerprint, likely strengths/weaknesses, scenario behaviors, and embodied signal summary. `TwinService` stores the profile/persona and writes a profile to `FeedbackMemoryManager`.

### 5. Simulation options

**Benchmark mode** is the default. `ScenarioGenerator.generate_all()` creates 10 scenario records: three job interviews, three investor pitches, and four dates. Each record contains an archetype, opening prompt, curveball, closing prompt, and success criteria. The target personas are currently selected from static archetype lists and given variations; the code does not ask an LLM to generate 10 entirely new target personas at setup time.

**Targeted mode** uses the scenario description from the questionnaire. The UI calls `/twin/custom-persona` to generate a counter-party persona (optionally informed by generated scenario questions and answers), then starts `/simulation/begin` with `mode: "custom"`. A custom run uses the generated counter-party prompt and has a 50-response loop.

### 6. Simulation execution and grading

`POST /simulation/begin` verifies the twin and starts background work. `SimulationService` stores a running simulation record, then uses `SimulationLoop.run_batch()` for benchmark scenarios with `SIM_MAX_WORKERS` workers (default 5). The browser polls `GET /simulation/<sim_id>` and displays completed conversations and scores.

For each benchmark scenario, the active `run_single()` path executes 50 loop iterations. Each iteration generates one twin response; the first 49 iterations also generate a counter-party response. Stage labels progress from opening to reaction, curveball, pressure, and closing. A `RefereeAgent` then grades the conversation on alignment, friction, outcome, and overall score, and returns a success/partial-success/failure verdict plus key moments and improvement tags.

The simulation model name comes from `SIM_LLM_MODEL`, falling back to `LLM_MODEL`, then `gemini-2.5-pro`. The twin, counter-party, and referee use the configured Vertex AI client. Model calls are logged by `agent/token_logger.py`.

### Model and runtime configuration

| Setting | Used by | Current fallback / behavior |
|---|---|---|
| `COUNSELLOR_MODEL` | Video coaching report | Defaults to `gemini-2.5-pro` |
| `LLM_MODEL` | Twin persona, feedback/cluster analysis, default simulation model | Defaults to `gemini-2.5-pro` |
| `SIM_LLM_MODEL` | Twin dialogue, counter-party dialogue, referee | Falls back to `LLM_MODEL`, then `gemini-2.5-pro` |
| `VERTEX_PROJECT`, `VERTEX_LOCATION` | Gemini clients | Default to `ai-ml-integrations` and `us-central1` in these modules |
| `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CREDENTIALS_JSON`, `GCP_*`, `GOOGLE_API_KEY` | Google client authentication | `app.py` supports a credentials file, JSON environment variable, individual service-account fields, or API-key fallback |
| `SIM_MAX_WORKERS` | Concurrent benchmark scenarios | Defaults to `5` |
| `POSTGRES_URI`, `REDIS_URL` | Optional persistence, caching, and LangGraph memory | Missing or unavailable services fall back to in-process storage where implemented |

The Flask web experience starts from `app.py`. `main.py` is a separate command-line entry point that loads `config.json` (or a supplied config), analyzes a local video, prints the report to the terminal, and can save text output under `output/`.

### 7. Post-simulation coaching and memory

After the simulation reaches `completed`, the UI automatically calls `POST /analysis/<sim_id>`.

1. `AnalysisService` gets the scenario results and twin persona.
2. `FailureClusterAnalyzer` computes/organizes patterns and uses Deep Agents/LangGraph when available, with fallback paths when optional components are unavailable. Its memory tools can retrieve the twin profile, prior episodes, and similar failure notes.
3. `FeedbackGenerator` turns the cluster report into a user-facing debrief: aggregate success statistics, category verdicts, failure breakdown, personality insights, a priority behavior change, and a 30-day practice plan.
4. `AnalysisService` returns the report and starts background persistence/memory updates. `GET /insights/<user_id>` exposes aggregated past coaching insights to the authenticated owner.

## Live Digital Twin — Real-Time Call (`live-video` branch)

This is a separate, self-contained flow from the benchmark/custom simulation above. It needs no pre-recorded video and no questionnaire; it runs an actual live webcam/mic conversation, analyzed as it happens.

```mermaid
flowchart TD
    A[Log in] --> B[Pick a practice use case]
    B --> C[Describe the real person you're about to face]
    C --> D[The app asks 22-26 natural follow-up questions about that person]
    D --> E[User answers the follow-ups]
    E --> F[A persona of that person is built: personality, communication style, opening line]
    F --> G[Live call starts: webcam and microphone stream begin]
    G --> H[Every turn: the digital twin replies in character]
    G --> I[Every few seconds, in the background: facial expression, posture, and gesture are analyzed]
    G --> J[Every several seconds, in the background: voice tone, pace, and speech are analyzed]
    H --> K[User clicks End and Get Judged]
    I --> K
    J --> K
    K --> L[The real conversation is graded using the transcript and every measured signal]
    L --> M[The user's streak, growth history, and difficulty level are updated]
    M --> N[Coaching report shown: one-line judgment, Confidence Score, Growth Hexagon, moment-by-moment feedback]
```

### Phase by phase

**1. Use case selection.** Right after logging in, the user picks one of ten domain scenarios — job interviews, dating, a difficult conversation with a girlfriend or parents, regulatory audits, medical consultations, venture-capital pitching, crisis PR interviews, diplomatic negotiation, or corporate boardrooms. Each scenario card shows the user's current streak and difficulty level for that specific use case.

**2. Describing the person.** The user freely types who they are about to face — a recruiter, an investor, a date, a difficult boss. The app sends this description to Gemini, which responds with 22 to 26 natural, conversational follow-up questions about that specific person: their personality, communication style, attitude, values, pet peeves, decision-making style, body language tendencies, and what's at stake for them in this interaction. The questions are about the other person, not about the user.

**3. Building the persona.** Once the user answers those follow-ups, everything collected — the description, the answers, the chosen use case's context, and the user's current difficulty level — is sent to Gemini again, which designs a persona: a name, a role, a personality summary, a communication style, and a detailed instruction describing how that person should talk and react during a live, spoken conversation (short, natural turns, never breaking character). It also writes the very first line the persona will say when the call begins.

**4. The live conversation.** The webcam and microphone turn on. Every time the user speaks or types a reply, it is sent to the model, which answers back in character, subtly adjusted by a live read of how the user is coming across (for example, tense posture, a mostly neutral expression, or a fast speaking pace) so the persona reacts the way a real person subconsciously would.

**5. Live behavioral signal capture.** While the conversation is happening, the app is also quietly analyzing the user in the background:
   - Every few seconds, a video frame is captured and analyzed for facial expression and emotion, posture and openness, confidence signals, hand gesture activity, and eyebrow movement.
   - Every several seconds, a short audio clip is captured and analyzed for pitch, energy, speaking pace, pauses, and a running transcript of what was said.
   
   Both of these run in the background rather than blocking the conversation — if a frame takes too long to analyze, it's simply dropped in favor of a fresher one, since a slightly stale behavioral sample doesn't hurt the coaching average. Spoken audio is never dropped, since losing someone's actual words would be unacceptable.

**6. Ending the call and judging performance.** When the user clicks "End & Get Judged," a neutral judge model reviews the full transcript, every aggregated behavioral signal from the call, and a turn-by-turn breakdown of what the user's face, posture, and eyebrows were doing at each of their own turns. It produces a short written verdict on how well the user's approach fit the situation, a score for each of seven performance dimensions, a six-axis "growth" profile, specific moment-by-moment notes on what went right or wrong, and concrete coaching tips.

**7. Streak and progression.** Each completed call updates the user's streak for that specific use case: consecutive days practiced, a rolling history of their six-axis growth scores, and an automatic difficulty upgrade — from a patient, encouraging opponent, to a neutral and professional one, to a demanding, interruption-prone one that actively tests composure — once the user sustains a streak at a strong enough performance level. Reaching the hardest tier unlocks a use-case-specific badge (for example, "Boardroom Ready" or "Pitch Perfect").

### Scoring shown in the coaching report

- **One-line judgment summary:** a short, qualitative read from the judge on whether the user's overall approach fit the specific situation they described. It is shown as plain text, without a numeric score next to it — showing a second, differently-scaled number alongside the Confidence Score was confusing even after the two were made consistent with each other.
- **Vyakti Confidence Score (0–100):** a single composite number. Rather than trusting a second, independently generated number from the model (which was found to reliably confuse its own 0–100 scale with the 1–10 scale used for every other score, a real bug that has since been fixed), this score is now always calculated directly from the average of the seven individual performance-dimension scores, scaled up to 100.
- **Growth Hexagon (six axes):** composure under pressure, vocal tone and control, posture and openness, how well facial expression matched the conversation's tone, conciseness and clarity, and active-listening behavior — tracked over time, per use case, as part of the user's ongoing streak.

### How this holds up in production

- **Everything about an in-progress call lives only in server memory** while the call is happening — it does not survive a server restart, the same way an in-progress video-analysis job does not. A separate small file on disk does persist each user's streak history across restarts.
- **The server intentionally runs as a single process,** because conversation, job, and call state all live in that same in-memory fashion; running multiple processes would scatter a single user's session across them unpredictably.
- **Frame and audio analysis happen fully in the background**, not as part of the request that the browser is waiting on. Earlier, a single frame's full analysis (checking facial expression, posture, hands, and eyebrows together) could take anywhere from several seconds to several minutes under load — far longer than how often the browser was sending new frames — which blocked the shared pool of request-handling capacity badly enough that live chat replies would occasionally never arrive in time at all. Moving this analysis fully off the request path, and deliberately dropping a stale frame rather than queuing it, fixed that.
- **The underlying machine-learning libraries used for this analysis are explicitly limited to one thread each,** since by default they each try to use every available processor core on their own — running several of them at once, per request, on a modest server quickly overwhelmed it and made everything slower, not faster.
- **Responses are protected against a data-type serialization bug** where the facial-emotion analysis library occasionally returns numbers in a format the web framework's default response encoder cannot handle, which used to crash the very last step of every call (the final judgment) once that analysis path was actually working correctly. A safety net was added so this class of error can no longer take down a response.
- **Spoken replies from the digital twin are read aloud in the browser.** A known browser quirk can silently cut off that spoken audio partway through if nothing keeps it "alive" for its full duration; this is now guarded against, along with basic logging so a future failure of this kind is visible instead of silent.

## Data and Persistence Boundaries

| Data | Current location / behavior |
|---|---|
| Uploaded video and extracted audio | Temporary files; deleted when video analysis finishes or fails |
| Video job status/results and user context | Flask process memory (`jobs`, `user_contexts`); not durable across process restart |
| User accounts | PostgreSQL when `POSTGRES_URI` is set; otherwise process-memory dictionaries |
| Twins | PostgreSQL when configured; otherwise process-memory dictionary plus `output/twins_store.json` |
| Simulation records/results | PostgreSQL when configured; otherwise process memory; active/progress values also use `RedisCache` |
| Analysis documents | PostgreSQL when configured; otherwise process memory; cached by `RedisCache` |
| Feedback memory | Redis/Postgres LangGraph checkpointer when configured and available; otherwise `InMemorySaver`; long-term store uses `PostgresStore` or `InMemoryStore` (optionally with embeddings) |
| Pinecone/Weaviate wrapper | Implemented in `agent/memory/vector_store.py`, but not currently constructed or called by the app's twin/feedback flow |
| Live Digital Twin sessions | Flask process memory only, keyed by a short session id; no Postgres/Redis path — not durable across a restart, same caveat as video jobs above |
| Vyakti Streak (live-call gamification) | A small JSON file on disk (loaded into memory at startup, rewritten on every recorded session) — the only live-call data that persists across a restart |

The short-term LangGraph checkpointer and long-term LangGraph store are distinct from the standalone `RedisCache` and the Pinecone/Weaviate wrapper. A Redis or PostgreSQL fallback in one of these components does not make process-local job state durable.

## Route Map

| Area | Routes used by the UI | Purpose |
|---|---|---|
| Video coach | `/init-job`, `/save-context`, `/upload-context-file`, `/upload`, `/status/<job_id>`, `/results/<job_id>` | Prepare context, upload, poll, and retrieve analysis |
| Auth | `/auth/register`, `/auth/login` | Register and authenticate |
| Twin | `/twin/schema`, `/twin/create`, `/twin/me`, `/twin/<twin_id>`, `/twin/update` | Load questionnaire and manage Digital Twin |
| Targeted preparation | `/twin/scenario-questions`, `/twin/custom-persona` | Generate scenario-specific questions and counter-party persona |
| Simulation | `/simulation/begin`, `/simulation/<sim_id>`, `/simulation/<sim_id>/step/<step>`, `/simulation/<sim_id>/results` | Start, poll, and read simulation outcomes |
| Coaching | `/analysis/<sim_id>`, `/analysis/get/<analysis_id>`, `/insights/<user_id>` | Generate/retrieve debrief and user insights |
| Live Digital Twin | `/use-cases`, `/streak/<use_case_id>`, `/streak`, `/live/questions`, `/live/build`, `/live/message`, `/live/frame`, `/live/audio`, `/live/end`, `/live/session/<session_id>` | Use-case/streak lookup, describe→follow-ups→build→live chat→judge for the real-time call |
| Diagnostics | `/admin/runs/<run_id>/telemetry`, `/admin/token-usage` | Inspect telemetry and model usage (authentication required) |

## Current Scope and Important Gaps

These distinctions matter when interpreting the UI and the existing architecture notes:

- **The benchmark is 10 scenarios, not 1,000.** Each benchmark scenario currently runs 50 twin response iterations (and 49 counter-party replies); it does not generate a thousand scenarios.
- **The four-stage LangGraph graph is not the benchmark execution path.** `SimulationLoop` builds an opening/reaction/curveball/closing graph, but `run_batch()` calls `run_single()`, which executes `_extended_run()` instead. The normal benchmark therefore follows the longer 50-iteration loop, including a pressure stage.
- **The target profiles are archetype templates.** The 10 benchmark counter-parties use static archetype data plus variations; the LLM generates their dialogue, not a new target-persona profile for each benchmark at scenario-generation time.
- **Pinecone/Weaviate evaluation is latency-only in the wrapper.** `VectorMemoryStore` can select the faster available service from a single latency check, but it does not measure retrieval accuracy, and current app services do not wire that class into the memory flow.
- **Dataset files are not used for training by the runtime.** The current application code does not load the `datasets/` contents into a training or inference pipeline. `download_coco.py` is a dataset download/export utility.
- **OpenPose is not the active pose analyzer.** The web analysis imports MediaPipe Tasks Pose and Hand Landmarkers. The `external_repos/openpose` folder is not imported in the application path.
- **Eye contact and micro-expression claims need care.** The questionnaire collects self-reported eye contact, but the video pipeline does not calculate gaze/eye-contact tracking. DeepFace classifies sampled-frame emotion; it is not a dedicated micro-expression model. Posture/spine/confidence values are landmark-derived proxies.
- **The browser contains draft and stop requests without matching Flask routes.** The UI calls `/twin/draft` and `/simulation/<sim_id>/stop`, but `app.py` does not register those endpoints; those actions currently cannot complete through the shown backend.
- **Job data is process-local and video routes are unauthenticated.** Do not assume a video job survives a restart or that `/upload` and `/results/<job_id>` enforce account ownership; only the twin/simulation/coaching route groups use the auth decorator in the current code.
- **Live Digital Twin sessions are process-local too, and require a single gunicorn worker.** `/live/*` session state lives in the same kind of in-memory dict as video jobs — it does not survive a restart, and the deployment is pinned to one worker process so this state isn't scattered across processes.
- **Live frame analysis intentionally drops frames under load.** `/live/frame` keeps only the most recently received frame per session and discards any that arrive while a prior one is still being analyzed — the coaching signal is a rolling average, not a frame-complete record of the call, by design.

## Main Code Map

| Concern | Main files |
|---|---|
| Flask routes, job lifecycle, model credential setup | `app.py` |
| Browser onboarding, video flows, questionnaire, simulation, report views | `templates/index.html` |
| Video decoding | `agent/video_processor.py` |
| Video analysis, weighted scores, behavioral profile | `agent/behavior_agent.py`, `agent/analyzers/` |
| Gemini video coach | `agent/analyzers/gemini_counsellor.py` |
| Twin questionnaire, profile, persona | `agent/twin/form_schema.py`, `agent/twin/profile_builder.py`, `agent/twin/persona_generator.py` |
| User, twin, simulation, feedback services | `services/user_service.py`, `services/twin_service.py`, `services/simulation_service.py`, `services/analysis_service.py` |
| Scenario generation, counter-parties, simulation, referee | `agent/simulation/scenario_generator.py`, `agent/simulation/counter_agents.py`, `agent/simulation/simulation_loop.py`, `agent/simulation/referee.py` |
| Failure analysis, feedback, and LangGraph memory | `agent/feedback/cluster_analyzer.py`, `agent/feedback/feedback_generator.py`, `agent/feedback/memory_manager.py` |
| Optional standalone cache/vector DB adapters | `agent/memory/redis_cache.py`, `agent/memory/vector_store.py` |
| Live Digital Twin: session orchestration, question/persona generation, live chat, judging, use cases, streaks | `agent/live/live_session_manager.py`, `agent/live/live_question_generator.py`, `agent/live/live_persona_builder.py`, `agent/live/live_twin_chat.py`, `agent/live/live_behavior_tracker.py`, `agent/live/live_judge.py`, `agent/live/use_cases.py`, `agent/live/streak_manager.py` |
