# HCA v2 — Current Architecture

## System Overview

HCA (Human Communication Agent) is an AI communication coach that:
1. Analyses a user's video (posture, facial expressions, voice)
2. Builds a **Digital Twin** persona from the analysis + a behavioural questionnaire
3. Runs **social simulations** (10 scenarios × 50 turns) against AI counter-parties
4. Scores each scenario and generates a personalised improvement plan

---

## LLM Call Summary

| Step | Caller | Model | Calls per run | Notes |
|---|---|---|---|---|
| Video counselling report | `GeminiCounsellor` | `gemini-3.5-flash` | **1** | Full JSON analysis report |
| Twin persona generation | `PersonaGenerator` | `gemini-3.5-flash` | **1** | Builds system_prompt from Big5/MBTI |
| Twin dialogue responses | `_build_twin_response()` | `gemini-2.5-flash` | **500** | 50 turns × 10 scenarios |
| Counter-party responses | `RecruiterAgent / InvestorAgent / DateAgent` | `gemini-2.5-flash` | **500** | 50 turns × 10 scenarios |
| Scenario grading | `RefereeAgent` | `gemini-2.5-flash` | **10** | 1 grading call per scenario |
| **Total per full run** | | | **~1,012 calls** | |

> Twin + counter-party calls run concurrently via `ThreadPoolExecutor(max_workers=5)` — 5 scenarios in parallel.

---

## End-to-End Request Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as index.html
    participant Flask as Flask app.py
    participant AS as AnalysisService
    participant BAA as BehaviorAnalysisAgent
    participant VP as VideoProcessor (MediaPipe)
    participant Analyzers as 3 Analyzers
    participant GC as GeminiCounsellor
    participant TS as TwinService
    participant PG as PersonaGenerator
    participant SS as SimulationService
    participant SG as ScenarioGenerator
    participant SL as SimulationLoop (LangGraph)
    participant CA as CounterAgents
    participant REF as RefereeAgent
    participant FE as FeedbackEngine
    participant Gemini35 as gemini-3.5-flash
    participant Gemini25 as gemini-2.5-flash

    User->>UI: Upload video + fill context form
    UI->>Flask: POST /init-job → POST /upload
    Flask->>AS: run_analysis(job_id, video_path) [background thread]

    AS->>BAA: analyze(video_path, context)
    BAA->>VP: extract frames + landmarks (MediaPipe)
    VP-->>BAA: pose_data, hand_data, frames

    par Parallel analysis
        BAA->>Analyzers: FacialExpressionAnalyzer.analyze()
        BAA->>Analyzers: BodyLanguageAnalyzer.analyze()
        BAA->>Analyzers: VoiceSpeechAnalyzer.analyze()
    end
    Analyzers-->>BAA: facial_results, body_results, voice_results

    BAA->>GC: counsel(all_results)
    GC->>Gemini35: generate_content() ← LLM CALL #1
    Gemini35-->>GC: JSON report
    GC-->>AS: counselling, predictions, improvement_plan

    AS-->>Flask: job complete
    UI->>Flask: GET /results/:job_id
    Flask-->>UI: Full analysis JSON

    User->>UI: Fill twin questionnaire + submit
    UI->>Flask: POST /twin/create
    Flask->>TS: create_twin(form_data, analysis_results)
    TS->>PG: generate(profile)
    PG->>Gemini35: generate_content() ← LLM CALL #2
    Gemini35-->>PG: system_prompt + persona JSON
    TS-->>Flask: twin_id
    Flask-->>UI: twin created

    User->>UI: Start simulation
    UI->>Flask: POST /simulation/begin
    Flask->>SS: run_simulation(twin_id)
    SS->>SG: generate() → 10 scenarios (3 job · 3 investor · 4 dating)

    loop 10 scenarios (5 concurrent via ThreadPoolExecutor)
        SS->>SL: run_single(scenario, max_turns=50)

        loop 50 dialogue turns per scenario
            SL->>SL: _build_twin_response()
            SL->>Gemini25: generate_content() ← LLM CALLS #3–502 (twin)
            Gemini25-->>SL: twin utterance

            SL->>CA: agent.respond()
            CA->>Gemini25: generate_content() ← LLM CALLS #503–1002 (counter)
            Gemini25-->>CA: counter-party utterance
        end

        SL->>REF: grade(scenario, conversation, twin_persona)
        REF->>Gemini25: generate_content() ← LLM CALLS #1003–1012 (referee)
        Note over REF,Gemini25: thinking_budget=0 (fast JSON output)
        Gemini25-->>REF: scores (alignment · friction · outcome · overall)
        REF-->>SL: grade dict
    end

    SL->>FE: FailureClusterAnalyzer + FeedbackGenerator
    FE-->>SS: clusters + coaching_tips
    SS-->>Flask: simulation complete
    UI->>Flask: GET /simulation/:id
    Flask-->>UI: All scenario results + scores + feedback
```

---

## Component Architecture

```mermaid
flowchart TB
    subgraph Browser["🌐 Browser — templates/index.html"]
        UI["Single Page App\nUpload video · Fill twin form\nPoll status · View results"]
    end

    subgraph Server["🖥️ Flask — app.py  (port 5004 dev / $PORT Render)"]
        direction TB
        MW["JWT Middleware\nrequire_auth()"]

        subgraph Routes["Routes"]
            R1["POST /init-job\nPOST /upload\nGET  /status/:job_id\nGET  /results/:job_id"]
            R2["POST /auth/register\nPOST /auth/login"]
            R3["GET  /twin/schema\nPOST /twin/create\nGET  /twin/:id · /twin/me\nPUT  /twin/update"]
            R4["POST /simulation/begin\nGET  /simulation/:id\nGET  /simulation/:id/step/:n\nGET  /simulation/:id/results"]
            R5["POST /analysis/:sim_id\nGET  /analysis/get/:id\nGET  /insights/:user_id"]
        end
    end

    subgraph Services["services/"]
        US["UserService\nJWT sign/verify\npassword hashing (hmac)"]
        TS["TwinService\nProfile CRUD\noutput/twins_store.json\n29 twins loaded on start"]
        SS["SimulationService\nThreadPoolExecutor\norchestrates sim runs"]
        AS["AnalysisService\nThreadPoolExecutor\norchestrates analysis runs"]
    end

    subgraph AnalysisPipeline["agent/ — Analysis Pipeline"]
        BAA["BehaviorAnalysisAgent\nbehavior_agent.py"]
        VP["VideoProcessor\nvideo_processor.py\nMediaPipe pose + hands\nOpenCV frame extraction\n[NO LLM]"]
        FA["FacialExpressionAnalyzer\nfacial_expression.py\n[NO LLM]"]
        BLA["BodyLanguageAnalyzer\nbody_language.py\n[NO LLM]"]
        VSA["VoiceSpeechAnalyzer\nvoice_speech.py\n[NO LLM]"]
        CI["UserContext\ncontext_intake.py\n[NO LLM]"]
        GC["🤖 GeminiCounsellor\ngemini_counsellor.py\n1 LLM call per video\ngemini-3.5-flash\n→ counselling, predictions,\nstrengths, improvement plan"]
    end

    subgraph TwinEngine["agent/twin/ — Digital Twin Engine"]
        PB["TwinProfileBuilder\nprofile_builder.py\nMaps analysis → Big5 / MBTI\n[NO LLM]"]
        PG["🤖 PersonaGenerator\npersona_generator.py\n1 LLM call per twin\ngemini-3.5-flash\n→ system_prompt + persona JSON"]
        FS["form_schema.py\nQuestionnaire schema\n(behavioural + cognitive traits)\n[NO LLM]"]
    end

    subgraph SimEngine["agent/simulation/ — Simulation Engine (LangGraph)"]
        SG["ScenarioGenerator\n10 scenarios per run:\n3 job · 3 investor · 4 dating\n[NO LLM]"]
        SL["🤖 SimulationLoop\nLangGraph StateGraph\n50 turns × 10 scenarios\nThreadPoolExecutor(5 parallel)\n500 LLM calls — twin responses\ngemini-2.5-flash"]
        CA["🤖 Counter Agents\nRecruiterAgent / InvestorAgent / DateAgent\n500 LLM calls — counter responses\ngemini-2.5-flash"]
        REF["🤖 RefereeAgent\nScores 1–10 per scenario\nalignment · friction · outcome\n10 LLM calls — 1 per scenario\ngemini-2.5-flash\nthinking_budget=0"]
    end

    subgraph FeedbackEngine["agent/feedback/ — Feedback Engine"]
        CLA["FailureClusterAnalyzer\ncluster_analyzer.py\nGroups failure patterns"]
        FG["FeedbackGenerator\nfeedback_generator.py\nActionable coaching tips"]
        MM["FeedbackMemoryManager\nmemory_manager.py\nInMemorySaver (dev)\nLangGraph checkpointer"]
    end

    subgraph Memory["agent/memory/ + agent/cache/"]
        RC["RedisCache\nredis_cache.py\n→ in-memory fallback\n(Redis not running locally)"]
        VS["VectorMemoryStore\nvector_store.py\nPinecone / Weaviate\n→ InMemoryStore fallback"]
        GCC["GeminiContextCache\ncache/gemini_cache.py\nVertex AI context caching"]
    end

    subgraph GCP["☁️ Google Cloud — Vertex AI"]
        CREDS["impersonated_service_account\nvertex-ai-dev-sa@ai-ml-integrations\nOAuth2 refresh token\nloaded via GOOGLE_CREDENTIALS_JSON env var"]
        M1["gemini-3.5-flash\nLLM_MODEL · location=global\n2 calls per full run\n(1 counsellor + 1 persona)"]
        M2["gemini-2.5-flash\nSIM_LLM_MODEL · location=global\n~1010 calls per full run\n(500 twin + 500 counter + 10 referee)"]
    end

    UI -->|video + form| MW
    MW --> R1 & R2 & R3 & R4 & R5

    R1 --> AS
    R2 --> US
    R3 --> TS
    R4 --> SS
    R5 --> AS

    TS --> PB & PG & FS
    SS --> SG --> SL
    SL --> CA & REF
    SL --> FeedbackEngine

    AS --> BAA
    BAA --> VP
    VP --> FA & BLA & VSA
    FA & BLA & VSA & CI --> GC
    GC --> R1

    GC --> M1
    PG --> M1
    SL & CA --> M2
    REF --> M2
    M1 & M2 --> CREDS

    AS --> CLA & FG & MM
    MM --> RC & VS
    SL --> GCC
```

---

## Layer Reference

| Layer | Files | Responsibility |
|---|---|---|
| **Web UI** | `templates/index.html` | SPA — upload, form, polling, results display |
| **Flask App** | `app.py` | 18 REST endpoints, JWT auth, background threading |
| **Services** | `services/` | Business logic, orchestration, concurrency |
| **Analysis** | `agent/behavior_agent.py` + `agent/analyzers/` | Video → MediaPipe → 3 analyzers → Gemini JSON report |
| **Digital Twin** | `agent/twin/` | Big5/MBTI profile + LLM system_prompt generation |
| **Simulation** | `agent/simulation/` | LangGraph 50-turn multi-agent, 10 scenarios (3+3+4) |
| **Feedback** | `agent/feedback/` | Failure clustering, coaching tips, memory |
| **Memory** | `agent/memory/` + `agent/cache/` | Redis / Pinecone / Weaviate with in-memory fallbacks |
| **LLMs** | Vertex AI | `gemini-3.5-flash` (analysis) · `gemini-2.5-flash` (simulation) |

---

## Key Classes

| Class | File | Role |
|---|---|---|
| `BehaviorAnalysisAgent` | `agent/behavior_agent.py` | Orchestrates full analysis pipeline |
| `VideoProcessor` | `agent/video_processor.py` | Frame extraction, MediaPipe landmark detection |
| `FacialExpressionAnalyzer` | `agent/analyzers/facial_expression.py` | Micro-expression detection (anger, fear, joy…) |
| `BodyLanguageAnalyzer` | `agent/analyzers/body_language.py` | Posture, gesture, spine angle |
| `VoiceSpeechAnalyzer` | `agent/analyzers/voice_speech.py` | Pace, tone, filler words |
| `GeminiCounsellor` | `agent/analyzers/gemini_counsellor.py` | Final structured JSON report via `gemini-3.5-flash` |
| `TwinProfileBuilder` | `agent/twin/profile_builder.py` | Maps analysis results to Big5 / MBTI scores |
| `PersonaGenerator` | `agent/twin/persona_generator.py` | Generates LLM system_prompt for the digital twin |
| `ScenarioGenerator` | `agent/simulation/scenario_generator.py` | Builds 10 counter-party archetypes per simulation |
| `SimulationLoop` | `agent/simulation/simulation_loop.py` | LangGraph StateGraph, 50 turns, 5 concurrent workers |
| `RecruiterAgent` | `agent/simulation/counter_agents.py` | Job interview counter-party |
| `InvestorAgent` | `agent/simulation/counter_agents.py` | Investor pitch counter-party |
| `DateAgent` | `agent/simulation/counter_agents.py` | Dating scenario counter-party |
| `RefereeAgent` | `agent/simulation/referee.py` | Scores each scenario 1–10 (thinking disabled) |
| `FailureClusterAnalyzer` | `agent/feedback/cluster_analyzer.py` | Groups failure patterns across scenarios |
| `FeedbackGenerator` | `agent/feedback/feedback_generator.py` | Generates actionable coaching tips |
| `FeedbackMemoryManager` | `agent/feedback/memory_manager.py` | Short/long-term memory (InMemorySaver fallback) |
| `RedisCache` | `agent/memory/redis_cache.py` | Redis cache → in-memory fallback |
| `VectorMemoryStore` | `agent/memory/vector_store.py` | Pinecone / Weaviate → InMemoryStore fallback |
| `GeminiContextCache` | `agent/cache/gemini_cache.py` | Vertex AI context caching for simulation |
| `UserService` | `services/user_service.py` | User registration, JWT auth |
| `TwinService` | `services/twin_service.py` | Twin CRUD, disk persistence |
| `SimulationService` | `services/simulation_service.py` | Simulation lifecycle management |
| `AnalysisService` | `services/analysis_service.py` | Analysis job lifecycle management |

---

## Environment Variables

| Variable | Value | Purpose |
|---|---|---|
| `GOOGLE_CREDENTIALS_JSON` | Full JSON string | GCP impersonated_service_account credentials |
| `VERTEX_PROJECT` | `ai-ml-integrations` | GCP project |
| `VERTEX_LOCATION` | `global` | Required for `gemini-3.5-flash` |
| `LLM_MODEL` | `gemini-3.5-flash` | Model for analysis & persona generation |
| `SIM_LLM_MODEL` | `gemini-2.5-flash` | Model for simulation (twin, counter agents, referee) |
| `JWT_SECRET` | (random string) | JWT signing secret |
| `FLASK_ENV` | `production` / `development` | Flask environment |
| `PORT` | `5004` (default) | HTTP server port |
| `REDIS_URL` | *(optional)* | Redis — falls back to in-memory if absent |

---

## Deployment

| Environment | Host | Start Command |
|---|---|---|
| **Local dev** | `localhost:5004` | `python app.py` |
| **Production** | Render (`hca-1-6892.onrender.com`) | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120` |

> **Note:** Free tier on Render spins down after inactivity (~50s cold start).
> Redis and vector store fall back to in-memory — non-blocking in both environments.
