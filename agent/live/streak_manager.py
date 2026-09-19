"""
Vyakti Streak Manager
=======================
Implements Phase 2 of the Live Digital Twin architecture: a Use-Case-Specific
Matrix & Adaptive Progression Model, instead of a generic "days uploaded" streak.

- Domain-specific paths: one streak per (user_id, use_case_id) pair.
- Growth Hexagon: 6-axis rolling history (composure, vocal resonance, posture,
  facial congruence, conciseness, active listening) taken straight from the
  LivePerformanceJudge's "growth_hexagon" output.
- Algorithmic Difficulty Scaling: Level 1 (Novice) -> Level 2 (Neutral) ->
  Level 3 (Hostile/Mastery), auto-graduating on sustained daily streaks,
  awarding use-case-specific badges (e.g. "Boardroom Ready", "Crisis Proof").
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from agent.live.use_cases import get_use_case

_STORE_FILE = Path(__file__).parent.parent.parent / "output" / "vyakti_streaks.json"

# Consecutive-day thresholds to auto-graduate to the next difficulty level.
_LEVEL_UP_STREAK_DAYS = {1: 3, 2: 7}
_MAX_HISTORY = 60

_streaks: dict = {}


def _load():
    if _STORE_FILE.exists():
        try:
            _streaks.update(json.loads(_STORE_FILE.read_text()))
        except Exception as e:
            print(f"[VyaktiStreak] Could not load store: {e}")


_load()


def _save():
    try:
        _STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STORE_FILE.write_text(json.dumps(_streaks, default=str))
    except Exception as e:
        print(f"[VyaktiStreak] Could not persist store: {e}")


def _key(user_id: str, use_case_id: str) -> str:
    return f"{user_id}:{use_case_id}"


def _default_entry(user_id: str, use_case_id: str) -> dict:
    return {
        "user_id": user_id,
        "use_case_id": use_case_id,
        "current_streak": 0,
        "longest_streak": 0,
        "last_session_date": None,
        "level": 1,
        "history": [],
        "badges": [],
    }


class VyaktiStreakManager:
    """Tracks per-use-case streaks, growth-hexagon history, and difficulty level."""

    def get_status(self, user_id: str, use_case_id: str) -> dict:
        entry = _streaks.get(_key(user_id, use_case_id)) or _default_entry(user_id, use_case_id)
        return self._public_view(entry)

    def get_all_status(self, user_id: str) -> list:
        out = []
        for key, entry in _streaks.items():
            if entry.get("user_id") == user_id:
                out.append(self._public_view(entry))
        return out

    def record_session(self, user_id: str, use_case_id: str, judgment: dict) -> dict:
        """Called once per completed live session (after judging) to update the streak."""
        key = _key(user_id, use_case_id)
        entry = _streaks.get(key) or _default_entry(user_id, use_case_id)

        today = date.today()
        last_str = entry.get("last_session_date")
        last = date.fromisoformat(last_str) if last_str else None

        if last == today:
            pass  # already logged today — don't double count, but still refresh history below
        elif last == today - timedelta(days=1):
            entry["current_streak"] += 1
        else:
            entry["current_streak"] = 1
        entry["last_session_date"] = today.isoformat()
        entry["longest_streak"] = max(entry.get("longest_streak", 0), entry["current_streak"])

        hexagon = judgment.get("growth_hexagon", {}) or {}
        confidence_score = judgment.get("confidence_score", 0)
        entry.setdefault("history", []).append({
            "date": today.isoformat(),
            "confidence_score": confidence_score,
            "growth_hexagon": hexagon,
            "verdict": judgment.get("verdict"),
        })
        entry["history"] = entry["history"][-_MAX_HISTORY:]

        newly_earned_badge = self._maybe_level_up(entry, use_case_id)

        _streaks[key] = entry
        _save()

        result = self._public_view(entry)
        result["newly_earned_badge"] = newly_earned_badge
        return result

    def _maybe_level_up(self, entry: dict, use_case_id: str) -> Optional[str]:
        level = entry.get("level", 1)
        threshold = _LEVEL_UP_STREAK_DAYS.get(level)
        if threshold is None:
            return None  # already at max level (3)
        if entry["current_streak"] < threshold:
            return None
        # Require decent performance, not just showing up, to graduate.
        recent = entry["history"][-threshold:]
        avg_conf = sum(h.get("confidence_score", 0) for h in recent) / len(recent) if recent else 0
        if avg_conf < 55:
            return None

        entry["level"] = level + 1
        if entry["level"] == 3:
            uc = get_use_case(use_case_id) or {}
            badge = uc.get("badge_level3", "Mastery Achieved")
            if badge not in entry.get("badges", []):
                entry.setdefault("badges", []).append(badge)
                return badge
        return None

    def _public_view(self, entry: dict) -> dict:
        from agent.live.use_cases import get_twin_level
        level = entry.get("level", 1)
        return {
            "use_case_id": entry.get("use_case_id"),
            "current_streak": entry.get("current_streak", 0),
            "longest_streak": entry.get("longest_streak", 0),
            "last_session_date": entry.get("last_session_date"),
            "level": level,
            "level_name": get_twin_level(level)["name"],
            "history": entry.get("history", []),
            "badges": entry.get("badges", []),
        }


_manager: Optional[VyaktiStreakManager] = None


def get_streak_manager() -> VyaktiStreakManager:
    global _manager
    if _manager is None:
        _manager = VyaktiStreakManager()
    return _manager
