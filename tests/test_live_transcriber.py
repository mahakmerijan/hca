import base64
import io
import wave

import pytest

from agent.live import live_transcriber
from agent.live.live_transcriber import transcribe_audio_b64


def _silent_wav_b64(seconds=0.5, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return base64.b64encode(buf.getvalue()).decode()


def test_transcribes_with_recognizer(monkeypatch):
    calls = {}

    def fake_recognize(self, audio, language="en-US"):
        calls["language"] = language
        calls["bytes"] = len(audio.get_raw_data())
        return "hello there"

    monkeypatch.setattr(live_transcriber.sr.Recognizer, "recognize_google", fake_recognize)
    assert transcribe_audio_b64(_silent_wav_b64(), mimetype="audio/wav") == "hello there"
    assert calls["language"] == "en-US"
    assert calls["bytes"] > 0


def test_unintelligible_audio_returns_empty(monkeypatch):
    def fake_recognize(self, audio, language="en-US"):
        raise live_transcriber.sr.UnknownValueError()

    monkeypatch.setattr(live_transcriber.sr.Recognizer, "recognize_google", fake_recognize)
    assert transcribe_audio_b64(_silent_wav_b64(), mimetype="audio/wav") == ""


def test_service_error_is_raised(monkeypatch):
    def fake_recognize(self, audio, language="en-US"):
        raise live_transcriber.sr.RequestError("down")

    monkeypatch.setattr(live_transcriber.sr.Recognizer, "recognize_google", fake_recognize)
    with pytest.raises(live_transcriber.TranscriptionError):
        transcribe_audio_b64(_silent_wav_b64(), mimetype="audio/wav")


def test_invalid_audio_raises():
    with pytest.raises(live_transcriber.TranscriptionError):
        transcribe_audio_b64("not-base64!!", mimetype="audio/webm")
