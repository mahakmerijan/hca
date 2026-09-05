"""
GCP LLM Token Logger
====================
Wraps every generate_content call to track token usage with timestamps.
Writes to logs/token_usage.jsonl — one JSON line per call.
Also keeps an in-memory daily summary accessible via get_summary().
"""

import json
import os
import threading
import time
import tracemalloc
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, date
from pathlib import Path
from typing import Optional

_LOG_DIR = Path(os.getenv("HCA_TELEMETRY_LOG_DIR", Path(os.path.dirname(__file__)).parent / "logs"))
_LOG_FILE = _LOG_DIR / "token_usage.jsonl"
_lock = threading.Lock()
_run_context: ContextVar[dict] = ContextVar("run_context", default={})

if not tracemalloc.is_tracing():
    tracemalloc.start(25)

# In-memory daily totals: { "2026-07-01": {"input": int, "output": int, "calls": int, "cost_usd": float} }
_daily: dict = {}

# Pricing per 1M tokens (paid tier from Google AI pricing page)
_PRICING = {
    "gemini-2.5-flash":  {"input": 0.30, "output": 2.50},
    "gemini-3.5-flash":  {"input": 1.50, "output": 9.00},
    "gemini-2.5-pro":    {"input": 1.25, "output": 10.00},
}


def _ensure_log_dir():
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def _memory_snapshot() -> dict:
    """Return local-process memory measurements without external dependencies."""
    current, peak = tracemalloc.get_traced_memory()
    try:
        import resource
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KiB.
        if os.uname().sysname != "Darwin":
            peak_rss *= 1024
    except Exception:
        peak_rss = 0
    return {
        "python_allocated_bytes": current,
        "python_peak_bytes": peak,
        "process_peak_rss_bytes": peak_rss,
    }


def _memory_delta(before: dict, after: dict) -> dict:
    return {
        "python_allocated_delta_bytes": after["python_allocated_bytes"] - before["python_allocated_bytes"],
        "process_peak_rss_delta_bytes": after["process_peak_rss_bytes"] - before["process_peak_rss_bytes"],
    }


def _append_record(record: dict) -> dict:
    _ensure_log_dir()
    with _lock:
        try:
            with open(_LOG_FILE, "a") as log_file:
                log_file.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            print(f"[TokenLogger] Write error: {exc}")
    return record


@contextmanager
def run_context(run_id: str, run_type: str, user_id: str = ""):
    """Attach a run identity to every telemetry record on the current thread."""
    context = {"run_id": run_id, "run_type": run_type, "user_id": user_id}
    token = _run_context.set(context)
    _append_record({
        "_type": "run_started",
        "ts": datetime.utcnow().isoformat() + "Z",
        **context,
        **_memory_snapshot(),
    })
    try:
        yield
    finally:
        _append_record({
            "_type": "run_finished",
            "ts": datetime.utcnow().isoformat() + "Z",
            **context,
            **_memory_snapshot(),
        })
        _run_context.reset(token)


@contextmanager
def track_stage(component: str, model: str, model_kind: str, extra: Optional[dict] = None):
    """Record duration and memory for a local model or processing stage."""
    context = _run_context.get().copy()
    started = time.perf_counter()
    started_at = datetime.utcnow().isoformat() + "Z"
    before = _memory_snapshot()
    status = "completed"
    error = ""
    try:
        yield
    except Exception as exc:
        status = "error"
        error = str(exc)
        raise
    finally:
        after = _memory_snapshot()
        _append_record({
            "_type": "model_stage",
            "ts": datetime.utcnow().isoformat() + "Z",
            "started_at": started_at,
            "ended_at": datetime.utcnow().isoformat() + "Z",
            **context,
            "component": component,
            "model": model,
            "model_kind": model_kind,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "status": status,
            "error": error,
            "memory_before": before,
            "memory_after": after,
            "memory_delta": _memory_delta(before, after),
            **(extra or {}),
        })


def log_call(
    model: str,
    caller: str,
    input_tokens: int,
    output_tokens: int,
    user_id: Optional[str] = None,
    sim_id: Optional[str] = None,
    extra: Optional[dict] = None,
):
    """
    Record a single LLM call.

    Args:
        model:         e.g. "gemini-2.5-flash"
        caller:        e.g. "RefereeAgent", "GeminiCounsellor", "SimulationLoop/twin"
        input_tokens:  token count from usage_metadata.prompt_token_count
        output_tokens: token count from usage_metadata.candidates_token_count
        user_id:       optional user context
        sim_id:        optional simulation context
        extra:         any additional fields to log
    """
    _ensure_log_dir()
    context = _run_context.get()

    prices = _PRICING.get(model, {"input": 0.0, "output": 0.0})
    cost = (input_tokens / 1_000_000 * prices["input"] +
            output_tokens / 1_000_000 * prices["output"])

    today = date.today().isoformat()
    record = {
        "ts":            datetime.utcnow().isoformat() + "Z",
        "date":          today,
        "model":         model,
        "caller":        caller,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "total_tokens":  input_tokens + output_tokens,
        "cost_usd":      round(cost, 6),
        "user_id":       user_id or context.get("user_id", ""),
        "sim_id":        sim_id or "",
        "run_id":        context.get("run_id", ""),
        "run_type":      context.get("run_type", ""),
        "model_kind":    "llm",
        "memory":        _memory_snapshot(),
    }
    if extra:
        record.update(extra)

    with _lock:
        # Update daily in-memory summary
        d = _daily.setdefault(today, {"input": 0, "output": 0, "calls": 0, "cost_usd": 0.0})
        d["input"]    += input_tokens
        d["output"]   += output_tokens
        d["calls"]    += 1
        d["cost_usd"] = round(d["cost_usd"] + cost, 6)

    return _append_record(record)


def extract_token_counts(response) -> tuple[int, int]:
    """
    Extract (input_tokens, output_tokens) from a Gemini SDK response object.
    Falls back to (0, 0) if usage_metadata is unavailable.
    """
    try:
        meta = response.usage_metadata
        inp = getattr(meta, "prompt_token_count", 0) or 0
        out = getattr(meta, "candidates_token_count", 0) or 0
        return int(inp), int(out)
    except Exception:
        return 0, 0


def log_run_summary(
    run_type: str,
    sim_id: Optional[str] = None,
    user_id: Optional[str] = None,
    started_at: Optional[str] = None,
):
    """
    Write a summary line to the log aggregating all calls for a given sim_id (or
    the current day if sim_id is None).  Call this at the end of each simulation
    run or counselling session.

    Args:
        run_type:   e.g. "simulation", "counselling", "twin_creation"
        sim_id:     the simulation/job ID to aggregate
        user_id:    optional user context
        started_at: ISO timestamp when the run started (for duration calc)
    """
    _ensure_log_dir()
    now_ts = datetime.utcnow().isoformat() + "Z"
    today = date.today().isoformat()

    # Aggregate from JSONL log for this sim_id
    total_input = total_output = total_calls = 0
    total_cost = 0.0
    callers: dict = {}

    try:
        with open(_LOG_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    # Skip summary lines themselves
                    if r.get("_type") == "run_summary":
                        continue
                    # Filter by sim_id if given, else today's date
                    if sim_id:
                        if r.get("sim_id") != sim_id:
                            continue
                    else:
                        if r.get("date") != today:
                            continue
                    total_input  += r.get("input_tokens", 0) or 0
                    total_output += r.get("output_tokens", 0) or 0
                    total_cost   += r.get("cost_usd", 0.0) or 0.0
                    total_calls  += 1
                    caller = r.get("caller", "unknown")
                    callers[caller] = callers.get(caller, 0) + 1
                except Exception:
                    pass
    except FileNotFoundError:
        pass

    duration_s = None
    if started_at:
        try:
            from datetime import timezone
            start = datetime.fromisoformat(started_at.rstrip("Z")).replace(tzinfo=timezone.utc)
            end   = datetime.fromisoformat(now_ts.rstrip("Z")).replace(tzinfo=timezone.utc)
            duration_s = round((end - start).total_seconds(), 1)
        except Exception:
            pass

    summary = {
        "_type":         "run_summary",
        "ts":            now_ts,
        "date":          today,
        "run_type":      run_type,
        "sim_id":        sim_id or "",
        "user_id":       user_id or "",
        "total_calls":   total_calls,
        "input_tokens":  total_input,
        "output_tokens": total_output,
        "total_tokens":  total_input + total_output,
        "cost_usd":      round(total_cost, 6),
        "callers":       callers,
        "duration_s":    duration_s,
        "started_at":    started_at or "",
        "ended_at":      now_ts,
    }

    with _lock:
        try:
            with open(_LOG_FILE, "a") as f:
                f.write(json.dumps(summary) + "\n")
        except Exception as e:
            print(f"[TokenLogger] Summary write error: {e}")

    print(
        f"[TokenLogger] Run summary [{run_type}] sim={sim_id or 'n/a'} "
        f"calls={total_calls} tokens={total_input+total_output:,} "
        f"cost=${total_cost:.4f}" + (f" duration={duration_s}s" if duration_s else ""),
        flush=True,
    )
    return summary


def get_summary(days: int = 7) -> dict:
    """Return per-day token and cost summary for the last N days."""
    from datetime import timedelta
    today = date.today()
    result = {}
    for i in range(days):
        d = (today - timedelta(days=i)).isoformat()
        result[d] = _daily.get(d, {"input": 0, "output": 0, "calls": 0, "cost_usd": 0.0})
    return result


def get_run_summaries(limit: int = 20) -> list:
    """Return the last N run_summary lines from the log, newest first."""
    summaries = []
    try:
        with open(_LOG_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("_type") == "run_summary":
                        summaries.append(r)
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return list(reversed(summaries[-limit:]))


def get_log_file_path() -> str:
    return str(_LOG_FILE)


def get_run_telemetry(run_id: str) -> list[dict]:
    """Return the complete local telemetry trace for exactly one run."""
    events = []
    try:
        with open(_LOG_FILE) as log_file:
            for line in log_file:
                try:
                    event = json.loads(line)
                    if event.get("run_id") == run_id:
                        events.append(event)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return events


def get_current_run_context() -> dict:
    """Return a copy of the current thread's telemetry context for worker propagation."""
    return _run_context.get().copy()


def print_job_token_summary(job_id: str) -> None:
    """
    Print a formatted token + cost breakdown for a specific job_id to stdout.
    Called at the end of each analysis job so you can see usage in the local console.
    """
    if not job_id:
        return

    rows: list[tuple[str, int, int, int, float]] = []  # (caller, calls, in, out, cost)
    caller_data: dict[str, dict] = {}

    try:
        with open(_LOG_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("_type") == "run_summary":
                        continue
                    if r.get("job_id") != job_id and r.get("extra", {}).get("job_id") != job_id:
                        # Also check nested extra.job_id
                        if str(r.get("job_id", "")) != job_id:
                            continue
                    caller = r.get("caller", "unknown")
                    d = caller_data.setdefault(caller, {"calls": 0, "inp": 0, "out": 0, "cost": 0.0})
                    d["calls"] += 1
                    d["inp"]   += r.get("input_tokens", 0) or 0
                    d["out"]   += r.get("output_tokens", 0) or 0
                    d["cost"]  += r.get("cost_usd", 0.0) or 0.0
                except Exception:
                    pass
    except FileNotFoundError:
        return

    if not caller_data:
        return

    # Build rows
    for caller, d in sorted(caller_data.items()):
        rows.append((caller, d["calls"], d["inp"], d["out"], d["cost"]))

    total_calls = sum(r[1] for r in rows)
    total_inp   = sum(r[2] for r in rows)
    total_out   = sum(r[3] for r in rows)
    total_cost  = sum(r[4] for r in rows)

    # Column widths
    col_w = [max(len(r[0]) for r in rows) + 2, 6, 10, 10, 10]
    col_w[0] = max(col_w[0], 22)
    width = sum(col_w) + len(col_w) * 3 + 1

    sep   = "─" * width
    thick = "═" * width

    lines = [
        "",
        f"╔{thick}╗",
        f"║  TOKEN USAGE — Job {job_id:<{width - 22}}║",
        f"╠{thick}╣",
        f"║  {'Caller':<{col_w[0]}} {'Calls':>{col_w[1]}}  {'Input':>{col_w[2]}}  {'Output':>{col_w[3]}}  {'Cost (USD)':>{col_w[4]}}  ║",
        f"║  {sep}  ║",
    ]
    for caller, calls, inp, out, cost in rows:
        lines.append(
            f"║  {caller:<{col_w[0]}} {calls:>{col_w[1]}}  {inp:>{col_w[2]},}  {out:>{col_w[3]},}  ${cost:>{col_w[4]-1}.4f}  ║"
        )
    lines += [
        f"╠{thick}╣",
        f"║  {'TOTAL':<{col_w[0]}} {total_calls:>{col_w[1]}}  {total_inp:>{col_w[2]},}  {total_out:>{col_w[3]},}  ${total_cost:>{col_w[4]-1}.4f}  ║",
        f"╚{thick}╝",
        "",
    ]
    print("\n".join(lines), flush=True)

