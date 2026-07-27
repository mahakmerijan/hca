"""
Web UI for Human Behavior Analysis Agent
Flask application serving the upload page and analysis API.
Integrates Gemini AI as a personal counsellor.
"""

import gc
import itertools
import json
import os
import tempfile
import uuid
import threading
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Build GOOGLE_APPLICATION_CREDENTIALS from individual .env fields ──────────
# ── Option A: full JSON blob stored as GOOGLE_CREDENTIALS_JSON env var ───────
# On Render (or any host without a filesystem), paste the entire JSON content
# as a single env var named GOOGLE_CREDENTIALS_JSON and we write it to a temp file.
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    _creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()
    if _creds_json:
        try:
            _gcp_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
            _gcp_tmp.write(_creds_json)
            _gcp_tmp.close()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _gcp_tmp.name
            print(f"[Startup] GCP credentials loaded from GOOGLE_CREDENTIALS_JSON → {_gcp_tmp.name}")
        except Exception as _e:
            print(f"[Startup] Could not write GOOGLE_CREDENTIALS_JSON temp file: {_e}")

# ── Option B: individual GCP_* fields (legacy service_account type) ──────────
# All GCP service-account fields are stored as separate env vars (GCP_*).
# We reconstruct the JSON and write it to a temp file so the Google SDK can use it.
# This works identically on localhost (.env) and Render (environment variables).
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    _private_key = os.getenv("GCP_PRIVATE_KEY", "").replace("\\n", "\n")
    if _private_key:
        _gcp_creds = {
            "type":                        os.getenv("GCP_TYPE", "service_account"),
            "project_id":                  os.getenv("GCP_PROJECT_ID", ""),
            "private_key_id":              os.getenv("GCP_PRIVATE_KEY_ID", ""),
            "private_key":                 _private_key,
            "client_email":                os.getenv("GCP_CLIENT_EMAIL", ""),
            "client_id":                   os.getenv("GCP_CLIENT_ID", ""),
            "auth_uri":                    os.getenv("GCP_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
            "token_uri":                   os.getenv("GCP_TOKEN_URI", "https://oauth2.googleapis.com/token"),
            "auth_provider_x509_cert_url": os.getenv("GCP_AUTH_PROVIDER_X509_CERT_URL", "https://www.googleapis.com/oauth2/v1/certs"),
            "client_x509_cert_url":        os.getenv("GCP_CLIENT_X509_CERT_URL", ""),
            "universe_domain":             os.getenv("GCP_UNIVERSE_DOMAIN", "googleapis.com"),
        }
        try:
            _gcp_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
            json.dump(_gcp_creds, _gcp_tmp)
            _gcp_tmp.close()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _gcp_tmp.name
            print(f"[Startup] GCP credentials loaded from env vars → {_gcp_tmp.name}")
        except Exception as _e:
            print(f"[Startup] Could not write GCP credentials temp file: {_e}")
# ─────────────────────────────────────────────────────────────────────────────

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from functools import wraps

print("[Startup] Importing agent modules...")
try:
    from agent.behavior_agent import BehaviorAnalysisAgent
    print("[Startup] BehaviorAnalysisAgent OK")
except Exception as _e:
    print(f"[Startup] BehaviorAnalysisAgent FAILED: {_e}")
    BehaviorAnalysisAgent = None

try:
    from agent.analyzers.gemini_counsellor import GeminiCounsellor
    print("[Startup] GeminiCounsellor OK")
except Exception as _e:
    print(f"[Startup] GeminiCounsellor FAILED: {_e}")
    GeminiCounsellor = None

try:
    from agent.analyzers.context_intake import UserContext, extract_text_from_file
    print("[Startup] context_intake OK")
except Exception as _e:
    print(f"[Startup] context_intake FAILED: {_e}")
    UserContext = None
    def extract_text_from_file(filename, data): return ""

from agent.twin.form_schema import TWIN_FORM_SCHEMA

# Lazy-import services to avoid circular deps at module load
_user_svc      = None
_twin_svc      = None
_sim_svc       = None
_analysis_svc  = None


def _services():
    global _user_svc, _twin_svc, _sim_svc, _analysis_svc
    if _user_svc is None:
        from services import UserService, TwinService, SimulationService, AnalysisService
        _user_svc     = UserService()
        _twin_svc     = TwinService()
        _sim_svc      = SimulationService()
        _analysis_svc = AnalysisService()
    return _user_svc, _twin_svc, _sim_svc, _analysis_svc


app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "flv", "wmv"}

# In-memory job store
jobs: dict = {}
# Per-job user context store
user_contexts: dict = {}

# Initialise the Gemini counsellor in a background thread so Flask starts immediately
gemini_counsellor = None

def _init_gemini():
    global gemini_counsellor
    if GeminiCounsellor is None:
        return
    try:
        gemini_counsellor = GeminiCounsellor()
    except Exception as _gc_err:
        print(f"[Startup] GeminiCounsellor init failed: {_gc_err}")

threading.Thread(target=_init_gemini, daemon=True).start()


# ── JWT auth decorator ────────────────────────────────────────────
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
        if not token:
            return jsonify({"error": "Missing auth token"}), 401
        user_svc, *_ = _services()
        user_id = user_svc.verify_token(token)
        if not user_id:
            return jsonify({"error": "Invalid or expired token"}), 401
        request.user_id = user_id
        return f(*args, **kwargs)
    return decorated


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_default_config(video_path: str) -> dict:
    """Build a config dict pointing at the uploaded video."""
    return {
        "video_path": video_path,
        "analysis": {
            "frame_sample_rate": 90,  # every 3 s at 30 fps — faster on low-CPU hosts
            "face_confidence_threshold": 0.5,
            "pose_confidence_threshold": 0.5,
            "audio_segment_duration": 5,
        },
        "weights": {
            "job_interview": {
                "facial_expression": 0.25,
                "body_language": 0.25,
                "voice_tone": 0.20,
                "speech_clarity": 0.15,
                "confidence_level": 0.15,
            },
            "business_deal": {
                "facial_expression": 0.20,
                "body_language": 0.20,
                "voice_tone": 0.25,
                "speech_clarity": 0.20,
                "confidence_level": 0.15,
            },
            "date": {
                "facial_expression": 0.30,
                "body_language": 0.25,
                "voice_tone": 0.20,
                "speech_clarity": 0.10,
                "confidence_level": 0.15,
            },
        },
        "output": {
            "save_report": False,
        },
    }


def _build_conversation_gists(simulation_results: list, limit: int = 5) -> list:
    """Create concise conversation snippets for final report/PDF."""
    gists = []
    for idx, result in enumerate(simulation_results[:limit]):
        conv = result.get("conversation", []) if isinstance(result, dict) else []
        twin_line = ""
        agent_line = ""
        for msg in conv:
            if msg.get("role") == "twin" and not twin_line:
                twin_line = str(msg.get("content", "")).strip()
            if msg.get("role") == "agent" and not agent_line:
                agent_line = str(msg.get("content", "")).strip()
            if twin_line and agent_line:
                break
        gists.append({
            "scenario_num": idx + 1,
            "category": result.get("category", ""),
            "score": result.get("total_score", 0),
            "outcome": result.get("outcome", ""),
            "twin_line": twin_line[:220],
            "agent_line": agent_line[:220],
        })
    return gists


def _merge_video_analyses(analyses: list) -> dict:
    """
    Merge analysis results from multiple video uploads into a single combined result.
    Averages numeric metrics; concatenates transcripts; keeps the latest counselling report.
    """
    if not analyses:
        return {}
    if len(analyses) == 1:
        return analyses[0]

    import copy

    merged = copy.deepcopy(analyses[0])

    def _avg(dicts, key, subkey, default=0):
        vals = [d.get(key, {}).get(subkey, default) for d in dicts if isinstance(d.get(key), dict)]
        return sum(vals) / len(vals) if vals else default

    # Average facial metrics
    fa_keys = ["smile_ratio", "total_frames_analyzed", "faces_detected"]
    for k in fa_keys:
        vals = [a.get("facial_analysis", {}).get(k, 0) for a in analyses]
        if any(v != 0 for v in vals):
            merged.setdefault("facial_analysis", {})[k] = sum(vals) / len(vals)

    # Average scenario scores across all analyses
    for section in ("facial_analysis", "body_language_analysis", "voice_speech_analysis"):
        scores = {}
        for a in analyses:
            for cat, score in (a.get(section, {}).get("scenario_scores") or {}).items():
                scores.setdefault(cat, []).append(score)
        if scores:
            merged.setdefault(section, {})["scenario_scores"] = {
                k: round(sum(v) / len(v), 2) for k, v in scores.items()
            }

    # Concatenate voice transcripts
    transcripts = []
    for a in analyses:
        vs = a.get("voice_speech_analysis", {})
        t = vs.get("full_transcript") or vs.get("audio_features", {}).get("speech_content", {}).get("transcript_preview", "")
        if t:
            transcripts.append(t)
    if transcripts:
        merged.setdefault("voice_speech_analysis", {})["full_transcript"] = " | ".join(transcripts)

    # Average final predictions
    for scenario in ("job_interview", "business_deal", "date"):
        probs = [a.get("final_predictions", {}).get(scenario, {}).get("probability", 0) for a in analyses]
        if any(p != 0 for p in probs):
            avg_prob = round(sum(probs) / len(probs), 1)
            merged.setdefault("final_predictions", {}).setdefault(scenario, {})["probability"] = avg_prob

    # Keep counselling from the last analysis (most recent video)
    if analyses[-1].get("counselling"):
        merged["counselling"] = analyses[-1]["counselling"]

    merged["_merged_from"] = len(analyses)
    return merged


def _run_analysis(job_id: str, video_path: str):
    """Background worker — runs full analysis pipeline then Gemini counselling."""
    user_context = user_contexts.get(job_id)
    audio_tmp_path = None
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = "Loading video…"

        config = get_default_config(video_path)
        agent = BehaviorAnalysisAgent(config)

        agent.results["timestamp"] = datetime.now().isoformat()

        # Step 1 – load video
        jobs[job_id]["progress"] = "Loading video…"
        agent.video_processor = __import__(
            "agent.video_processor", fromlist=["VideoProcessor"]
        ).VideoProcessor(video_path, agent.frame_sample_rate)
        agent.results["video_info"] = agent.video_processor.get_video_info()

        # Step 2 – estimate sampled frames
        jobs[job_id]["progress"] = "Preparing video frames…"
        sampled_frame_count = agent.video_processor.get_sampled_frame_count()

        # Step 3 – extract audio into a temp file (no uploads/ directory needed)
        jobs[job_id]["progress"] = "Extracting audio…"
        audio_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_tmp_path = audio_tmp.name
        audio_tmp.close()
        audio_path_result = agent.video_processor.extract_audio(audio_tmp_path)

        # Step 4 – facial expressions (capped at 60 frames to bound runtime)
        jobs[job_id]["progress"] = "Analyzing facial expressions…"
        from agent.analyzers.facial_expression import FacialExpressionAnalyzer
        agent.facial_analyzer = FacialExpressionAnalyzer(agent.face_confidence)
        for idx, frame in itertools.islice(agent.video_processor.iter_frames(), 60):
            agent.facial_analyzer.analyze_frame(frame, idx)
        agent.results["facial_analysis"] = agent.facial_analyzer.get_summary()
        agent.facial_analyzer = None
        gc.collect()

        # Step 5 – body language (capped at 30 frames — MediaPipe is heavy on low-CPU)
        jobs[job_id]["progress"] = "Analyzing body language…"
        from agent.analyzers.body_language import BodyLanguageAnalyzer
        agent.body_analyzer = BodyLanguageAnalyzer(agent.pose_confidence)
        for idx, frame in itertools.islice(agent.video_processor.iter_frames(), 30):
            agent.body_analyzer.analyze_frame(frame, idx)
        agent.results["body_language_analysis"] = agent.body_analyzer.get_summary()
        agent.body_analyzer.release()
        agent.body_analyzer = None
        gc.collect()

        # Step 6 – voice / speech
        jobs[job_id]["progress"] = "Analyzing voice & speech…"
        from agent.analyzers.voice_speech import VoiceSpeechAnalyzer
        agent.voice_analyzer = VoiceSpeechAnalyzer(agent.audio_segment_duration)
        if audio_path_result:
            agent.voice_analyzer.run_full_analysis(audio_path_result)
        agent.results["voice_speech_analysis"] = agent.voice_analyzer.get_summary()
        agent.voice_analyzer = None
        gc.collect()

        # Step 7 – compute weighted predictions & basic profile
        jobs[job_id]["progress"] = "Computing predictions…"
        agent._compute_final_predictions()
        agent._build_behavioral_profile()

        # Step 8 – Gemini AI Counsellor
        jobs[job_id]["progress"] = "🧠 AI Counsellor is analysing your behaviour…"
        if gemini_counsellor is not None:
            counselling = gemini_counsellor.generate_counselling(agent.results, user_context)
        else:
            counselling = {"error": "AI counsellor unavailable at startup"}
        agent.results["counselling"] = counselling

        # Cleanup video processor (analyzers already released above)
        if agent.body_analyzer:  # fallback in case of early-exit path
            agent.body_analyzer.release()
        if agent.video_processor:
            agent.video_processor.release()
        gc.collect()

        # Store results
        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = "Complete"
        jobs[job_id]["results"] = json.loads(json.dumps(agent.results, default=str))

    except Exception as exc:
        import traceback
        traceback.print_exc()
        jobs[job_id]["status"] = "error"
        jobs[job_id]["progress"] = f"Error: {exc}"
        jobs[job_id]["error"] = str(exc)
    finally:
        # Delete temp files — frees disk regardless of success or failure
        for tmp_path in (video_path, audio_tmp_path):
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


# ──────────────────────────── Routes ────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Use: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    # Reuse pre-initialised job_id if provided (context was already saved to it)
    job_id = request.form.get("job_id") or uuid.uuid4().hex[:12]
    ext = file.filename.rsplit(".", 1)[1].lower()
    # Write to a named temp file — no permanent uploads/ directory needed
    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
    file.save(tmp_video.name)
    tmp_video.close()

    # Initialise job
    jobs[job_id] = {"status": "queued", "progress": "Queued…", "results": None, "error": None}

    # Kick off background analysis (temp file path is cleaned up inside the worker)
    t = threading.Thread(target=_run_analysis, args=(job_id, tmp_video.name), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status": job["status"],
        "progress": job["progress"],
    })


@app.route("/results/<job_id>")
def job_results(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] == "error":
        return jsonify({"error": job.get("error", "Unknown error")}), 500
    if job["status"] != "done":
        return jsonify({"error": "Analysis not complete yet"}), 202
    return jsonify(job["results"])


@app.route("/save-context", methods=["POST"])
def save_context():
    """Save text-based context (job desc, business idea, date partner) for a job."""
    data = request.get_json(force=True)
    job_id = data.get("job_id", "")
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    ctx = user_contexts.get(job_id, {})
    for field in ("job_description", "business_idea", "date_partner"):
        val = data.get(field, "")
        if val:
            ctx[field] = val.strip()
    user_contexts[job_id] = ctx
    return jsonify({"ok": True})


@app.route("/upload-context-file", methods=["POST"])
def upload_context_file():
    """Upload a PDF/DOCX/TXT file for a context field."""
    job_id = request.form.get("job_id", "")
    context_type = request.form.get("context_type", "")  # job | business | date
    if not job_id or not context_type:
        return jsonify({"error": "job_id and context_type required"}), 400
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    file_bytes = f.read()
    text = extract_text_from_file(f.filename, file_bytes)
    field_map = {"job": "job_description", "business": "business_idea", "date": "date_partner"}
    field = field_map.get(context_type, context_type)
    ctx = user_contexts.get(job_id, {})
    ctx[field] = text
    user_contexts[job_id] = ctx
    return jsonify({"ok": True, "extracted_chars": len(text)})


@app.route("/init-job", methods=["POST"])
def init_job():
    """Pre-create a job_id so context can be attached before video upload."""
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "pending", "progress": "Waiting for video…", "results": None, "error": None}
    user_contexts[job_id] = {}
    return jsonify({"job_id": job_id})


# ══════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/auth/register", methods=["POST"])
def auth_register():
    """POST { email, name, password } → { user_id, token }"""
    data = request.get_json(force=True)
    user_svc, *_ = _services()
    result = user_svc.register(
        email=data.get("email", ""),
        name=data.get("name", ""),
        password=data.get("password", ""),
    )
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result), 201


@app.route("/auth/login", methods=["POST"])
def auth_login():
    """POST { email, password } → { user_id, token }"""
    data = request.get_json(force=True)
    user_svc, *_ = _services()
    result = user_svc.login(
        email=data.get("email", ""),
        password=data.get("password", ""),
    )
    if "error" in result:
        return jsonify(result), 401
    return jsonify(result)


# ══════════════════════════════════════════════════════════════════
# DIGITAL TWIN ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/twin/schema", methods=["GET"])
def twin_schema():
    """Return the Digital Twin questionnaire schema (public)."""
    return jsonify(TWIN_FORM_SCHEMA)


@app.route("/twin/create", methods=["POST"])
@require_auth
def twin_create():
    """
    POST {
      name, form_data,
      video_job_id?,          # single video (legacy)
      video_job_ids?,         # list of video job_ids for multi-video (new)
      scenario_description?,  # what the user is preparing for
      scenario_answers?       # answers to the dynamic scenario questionnaire
    } → twin document
    """
    data          = request.get_json(force=True)
    user_svc, twin_svc, *_ = _services()
    user_id       = request.user_id
    user          = user_svc.get_user(user_id)
    user_name     = data.get("name") or (user.get("name") if user else "User")

    # ── Collect video analysis (supports multiple videos) ────────────
    video_job_ids = data.get("video_job_ids") or []
    if data.get("video_job_id"):
        video_job_ids = list({data["video_job_id"]} | set(video_job_ids))

    completed_analyses = [
        jobs[jid]["results"]
        for jid in video_job_ids
        if jid in jobs and jobs[jid].get("status") == "done" and jobs[jid].get("results")
    ]

    # Fallback: use the most recently completed job if none specified
    if not completed_analyses:
        for jid, j in reversed(list(jobs.items())):
            if j.get("status") == "done" and j.get("results"):
                completed_analyses = [j["results"]]
                print(f"[TwinCreate] Falling back to video job {jid}")
                break

    video_analysis = _merge_video_analyses(completed_analyses) if completed_analyses else None

    form_data = data.get("form_data", {})
    answered_count = sum(
        1
        for v in form_data.values()
        if v not in (None, "") and (not isinstance(v, list) or len(v) > 0)
    )

    if answered_count < 8:
        return jsonify({
            "error": "Please complete the questionnaire first (at least basic sections)."
        }), 400

    if not video_analysis:
        return jsonify({
            "error": "Please upload and finish video analysis before creating your Digital Twin."
        }), 400

    # Attach scenario preparation context if provided
    scenario_description = data.get("scenario_description", "")
    scenario_answers     = data.get("scenario_answers", {})
    if scenario_description:
        form_data["_scenario_description"] = scenario_description
        form_data["_scenario_answers"]     = scenario_answers

    result = twin_svc.create_twin(
        user_id=user_id,
        user_name=user_name,
        form_data=form_data,
        video_analysis=video_analysis,
    )
    # Link twin to user record
    user_svc.set_twin_id(user_id, result["twin_id"])
    return jsonify(result), 201


@app.route("/twin/<twin_id>", methods=["GET"])
@require_auth
def twin_get(twin_id):
    """GET /twin/<twin_id> → twin document."""
    _, twin_svc, *_ = _services()
    twin = twin_svc.get_twin(twin_id)
    if not twin:
        return jsonify({"error": "Twin not found"}), 404
    return jsonify(twin)


@app.route("/twin/me", methods=["GET"])
@require_auth
def twin_me():
    """Return the authenticated user's latest twin."""
    _, twin_svc, *_ = _services()
    twin = twin_svc.get_twin_for_user(request.user_id)
    if not twin:
        return jsonify({"error": "No twin found for this user"}), 404
    return jsonify(twin)


@app.route("/twin/update", methods=["PUT"])
@require_auth
def twin_update():
    """PUT { twin_id, form_data, video_job_id? } → updated twin."""
    data = request.get_json(force=True)
    _, twin_svc, *_ = _services()

    video_analysis = None
    video_job_id   = data.get("video_job_id")
    if video_job_id and video_job_id in jobs and jobs[video_job_id].get("status") == "done":
        video_analysis = jobs[video_job_id].get("results")

    result = twin_svc.update_twin(
        twin_id=data.get("twin_id", ""),
        form_data=data.get("form_data", {}),
        video_analysis=video_analysis,
    )
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


# ══════════════════════════════════════════════════════════════════
# SCENARIO PREPARATION ROUTES (new)
# ══════════════════════════════════════════════════════════════════

@app.route("/twin/scenario-questions", methods=["POST"])
@require_auth
def scenario_questions():
    """
    POST { scenario_description: "I am preparing for a Series A pitch with a16z..." }
    → [ { id, label, question, type, options?, ... }, ... ]

    Generates a dynamic set of 8-12 targeted questions about the user's specific
    preparation context, to be answered before custom persona creation.
    """
    data = request.get_json(force=True)
    description = data.get("scenario_description", "").strip()
    if not description:
        return jsonify({"error": "scenario_description is required"}), 400

    from agent.twin.scenario_questionnaire_generator import ScenarioQuestionnaireGenerator
    gen = ScenarioQuestionnaireGenerator()
    questions = gen.generate_questions(description, user_id=request.user_id)
    return jsonify({"questions": questions})


@app.route("/twin/custom-persona", methods=["POST"])
@require_auth
def custom_persona_create():
    """
    POST {
      description: "A senior partner at McKinsey who is very data-driven...",
      questionnaire_answers: { "sq_stakes": 9, "sq_goal": "get an offer", ... }
    }
    → custom persona dict (same shape as built-in archetypes)

    Creates a simulation-ready counter-party persona from the user's description.
    Store the result and pass it as custom_persona to /simulation/begin.
    """
    data = request.get_json(force=True)
    description = data.get("description", "").strip()
    if not description:
        return jsonify({"error": "description is required"}), 400

    from agent.simulation.custom_persona_generator import CustomPersonaGenerator
    gen = CustomPersonaGenerator()
    persona = gen.generate(
        description=description,
        questionnaire_answers=data.get("questionnaire_answers", {}),
        user_id=request.user_id,
    )
    return jsonify(persona)


# ══════════════════════════════════════════════════════════════════
# SIMULATION ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/simulation/begin", methods=["POST"])
@require_auth
def simulation_start():
    """
    POST {
      twin_id,
      mode?: "predefined" | "custom" | "both"  (default "predefined"),
      custom_persona?:  persona dict from /twin/custom-persona
    } → { sim_id, status, total, mode }
    Launches a background simulation run.
    """
    data = request.get_json(force=True)
    _, twin_svc, sim_svc, _ = _services()
    twin_id = data.get("twin_id")
    if not twin_id:
        return jsonify({"error": "twin_id required"}), 400
    twin = twin_svc.get_twin(twin_id)
    if not twin:
        return jsonify({"error": "Twin not found"}), 404

    mode           = data.get("mode", "predefined")
    custom_persona = data.get("custom_persona")

    if mode not in ("predefined", "custom", "both"):
        return jsonify({"error": "mode must be 'predefined', 'custom', or 'both'"}), 400
    if mode in ("custom", "both") and not custom_persona:
        return jsonify({"error": "custom_persona is required when mode includes 'custom'"}), 400

    result = sim_svc.start_simulation(
        user_id=request.user_id,
        twin_id=twin_id,
        twin_persona=twin.get("persona", {}),
        mode=mode,
        custom_persona=custom_persona,
    )
    return jsonify(result), 202


@app.route("/simulation/<sim_id>", methods=["GET"])
@require_auth
def simulation_get(sim_id):
    """GET /simulation/<sim_id> → { sim_id, status, completed, total, live_turn, custom_result, ... }"""
    _, _, sim_svc, _ = _services()
    sim = sim_svc.get_simulation(sim_id)
    if not sim:
        return jsonify({"error": "Simulation not found"}), 404
    # Build all_turns from results so frontend can show all conversations
    results = sim.get("results", [])
    all_turns = []
    for i, r in enumerate(results):
        all_turns.append({
            "scenario_num":  i + 1,
            "total":         sim["total"],
            "category":      r.get("category", ""),
            "counter_party": r.get("counter_party_name", "Agent"),
            "score":         r.get("overall_score", 0),
            "outcome":       r.get("verdict", ""),
            "conversation":  r.get("conversation", []),
        })
    return jsonify({
        "sim_id":        sim["sim_id"],
        "status":        sim["status"],
        "mode":          sim.get("mode", "predefined"),
        "completed":     sim["completed"],
        "total":         sim["total"],
        "error":         sim.get("error"),
        "live_turn":     sim.get("live_turn"),
        "all_turns":     all_turns,
        "custom_result": sim.get("custom_result"),
    })


@app.route("/simulation/<sim_id>/step/<int:step>", methods=["GET"])
@require_auth
def simulation_step(sim_id, step):
    """GET /simulation/<sim_id>/step/<n> → single simulation step result."""
    _, _, sim_svc, _ = _services()
    result = sim_svc.get_simulation_step(sim_id, step)
    if result is None:
        return jsonify({"error": "Simulation not found"}), 404
    return jsonify(result)


@app.route("/simulation/<sim_id>/results", methods=["GET"])
@require_auth
def simulation_results_all(sim_id):
    """GET /simulation/<sim_id>/results → full results list (may be large)."""
    _, _, sim_svc, _ = _services()
    sim = sim_svc.get_simulation(sim_id)
    if not sim:
        return jsonify({"error": "Simulation not found"}), 404
    if sim["status"] not in ("completed", "error"):
        return jsonify({"error": "Simulation still running", "status": sim["status"]}), 202
    return jsonify(sim.get("results", []))


# ══════════════════════════════════════════════════════════════════
# ANALYSIS & INSIGHTS ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/analysis/<sim_id>", methods=["POST"])
@require_auth
def analysis_run(sim_id):
    """
    POST /analysis/<sim_id> — trigger cluster analysis on a completed simulation.
    Requires twin_id in body to fetch persona.
    Returns full analysis + coaching feedback.
    """
    data = request.get_json(force=True)
    _, twin_svc, sim_svc, analysis_svc = _services()

    sim = sim_svc.get_simulation(sim_id)
    if not sim:
        return jsonify({"error": "Simulation not found"}), 404
    if sim["status"] != "completed":
        return jsonify({"error": f"Simulation not completed (status: {sim['status']})"}), 400

    twin_id = data.get("twin_id") or sim.get("twin_id", "")
    twin    = twin_svc.get_twin(twin_id) if twin_id else None

    sim_results = sim_svc.get_results(sim_id)

    result = analysis_svc.run_analysis(
        sim_id=sim_id,
        simulation_results=sim_results,
        user_id=request.user_id,
        twin_persona=twin.get("persona", {}) if twin else {},
    )
    result["conversation_gists"] = _build_conversation_gists(sim_results, limit=5)
    return jsonify(result)


@app.route("/analysis/get/<analysis_id>", methods=["GET"])
@require_auth
def analysis_get(analysis_id):
    """GET /analysis/get/<analysis_id> → stored analysis document."""
    _, _, _, analysis_svc = _services()
    result = analysis_svc.get_analysis(analysis_id)
    if not result:
        return jsonify({"error": "Analysis not found"}), 404
    return jsonify(result)


@app.route("/insights/<user_id>", methods=["GET"])
@require_auth
def user_insights(user_id):
    """GET /insights/<user_id> → aggregated long-term coaching insights."""
    if request.user_id != user_id:
        return jsonify({"error": "Forbidden"}), 403
    _, _, _, analysis_svc = _services()
    return jsonify(analysis_svc.get_insights(user_id))


# ══════════════════════════════════════════════════════════════════
# ADMIN / OBSERVABILITY ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/admin/token-usage", methods=["GET"])
@require_auth
def admin_token_usage():
    """
    GET /admin/token-usage?days=7
    Returns per-day GCP LLM token and cost summary for the last N days.
    Also returns the log file path for manual inspection.

    Query params:
      days  — number of days to include (default 7, max 90)
    """
    from agent.token_logger import get_summary, get_log_file_path, get_run_summaries
    try:
        days = min(int(request.args.get("days", 7)), 90)
    except (ValueError, TypeError):
        days = 7

    summary   = get_summary(days=days)
    total_inp = sum(v["input"]    for v in summary.values())
    total_out = sum(v["output"]   for v in summary.values())
    total_cost= sum(v["cost_usd"] for v in summary.values())
    total_calls = sum(v["calls"]  for v in summary.values())

    run_summaries = get_run_summaries(limit=50)

    return jsonify({
        "days":        days,
        "per_day":     summary,
        "totals": {
            "input_tokens":  total_inp,
            "output_tokens": total_out,
            "total_tokens":  total_inp + total_out,
            "calls":         total_calls,
            "cost_usd":      round(total_cost, 4),
        },
        "run_summaries": run_summaries,
        "log_file": get_log_file_path(),
    })


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    port = int(os.environ.get("PORT", 5004))
    print(f"[Startup] Flask starting on http://localhost:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)
