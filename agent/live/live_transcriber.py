"""
Server-side speech-to-text for the live-call mic button.

The browser's built-in SpeechRecognition only works where the browser vendor
provides a speech backend (e.g. desktop Chrome). Elsewhere (Electron/VS Code's
browser, Firefox, some Chromium builds) it fails with a "network" error, so the
frontend records the user's reply and posts it here instead.
"""

import base64
import binascii
import os
import tempfile

import speech_recognition as sr

MAX_AUDIO_SECONDS = 60


class TranscriptionError(Exception):
    pass


def _suffix_for(mimetype: str) -> str:
    mimetype = (mimetype or "").lower()
    for key, suffix in (("wav", ".wav"), ("ogg", ".ogg"), ("mp4", ".mp4"), ("mpeg", ".mp3")):
        if key in mimetype:
            return suffix
    return ".webm"


def transcribe_audio_b64(audio_b64: str, mimetype: str = "", language: str = "en-US") -> str:
    """Return the transcript ("" if no speech was understood); raise TranscriptionError on failure."""
    try:
        raw = base64.b64decode(audio_b64 or "", validate=True)
    except (binascii.Error, ValueError) as e:
        raise TranscriptionError("Invalid audio data") from e
    if not raw:
        raise TranscriptionError("Empty audio")

    suffix = _suffix_for(mimetype)
    paths = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(raw)
        paths.append(f.name)
        wav_path = f.name
        if suffix != ".wav":
            from pydub import AudioSegment
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out:
                pass
            paths.append(out.name)
            AudioSegment.from_file(f.name).export(out.name, format="wav")
            wav_path = out.name

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source, duration=MAX_AUDIO_SECONDS)
        return (recognizer.recognize_google(audio, language=language) or "").strip()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        raise TranscriptionError(f"Speech service unavailable: {e}") from e
    except TranscriptionError:
        raise
    except Exception as e:
        raise TranscriptionError(f"Could not process audio: {e}") from e
    finally:
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass
