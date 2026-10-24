from agent.live.live_question_generator import (
    GENDER_QUESTION,
    LiveTwinQuestionGenerator,
    normalize_gender,
    with_gender_question,
)


def test_gender_question_is_first_required_choice():
    questions = with_gender_question([{"id": "q1", "question": "Who are they?", "required": True}])
    assert questions[0] == GENDER_QUESTION
    assert questions[0]["required"] is True
    assert questions[0]["type"] == "choice"
    assert questions[0]["options"] == ["Male", "Female"]
    assert [q["id"] for q in questions] == ["gender", "q1"]


def test_llm_gender_duplicates_are_removed():
    questions = with_gender_question([
        {"id": "gender", "question": "Gender?"},
        {"id": "q2", "question": "Is this person male or female?"},
        {"id": "q3", "question": "What's their gender?"},
        {"id": "q4", "question": "What do they value?"},
    ])
    assert [q["id"] for q in questions] == ["gender", "q4"]


def test_fallback_questions_include_gender_first():
    gen = LiveTwinQuestionGenerator.__new__(LiveTwinQuestionGenerator)
    gen.client = None
    assert gen.generate_questions("A recruiter")[0]["id"] == "gender"


def test_normalize_gender():
    assert normalize_gender("Male") == "male"
    assert normalize_gender(" FEMALE ") == "female"
    assert normalize_gender("other") is None
    assert normalize_gender(None) is None


def test_build_twin_returns_selected_gender(monkeypatch):
    from agent.live import live_session_manager as lsm

    mgr = lsm.LiveTwinSessionManager.__new__(lsm.LiveTwinSessionManager)

    class Builder:
        def build(self, *args, **kwargs):
            return {"persona_name": "Jordan", "gender": "male", "opening_line": "Hi"}

    mgr._persona_builder = Builder()
    lsm._sessions["s1"] = {"user_id": None, "use_case_id": None, "description": "d", "conversation": []}
    result = mgr.build_twin("s1", {"gender": "Female", "q1": "Recruiter"})
    assert result["persona"]["gender"] == "female"
    lsm._sessions.pop("s1")
