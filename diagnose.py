#!/usr/bin/env python3
"""
Run this to diagnose startup failures:
    python diagnose.py
"""
import sys, traceback

results = {}

def test(name, fn):
    try:
        fn()
        results[name] = "OK"
        print(f"  [OK]   {name}")
    except Exception as e:
        results[name] = f"FAIL: {e}"
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()

print("\n=== Python ===")
print(f"  Executable: {sys.executable}")
print(f"  Version:    {sys.version}")

print("\n=== Core dependencies ===")
test("flask",        lambda: __import__("flask"))
test("werkzeug",     lambda: __import__("werkzeug"))
test("dotenv",       lambda: __import__("dotenv"))
test("numpy",        lambda: __import__("numpy"))
test("rich",         lambda: __import__("rich"))

print("\n=== CV / ML ===")
test("cv2",          lambda: __import__("cv2"))
test("mediapipe",    lambda: __import__("mediapipe"))
test("mediapipe tasks", lambda: (
    __import__("mediapipe").tasks.vision.PoseLandmarker
))
test("moviepy",      lambda: __import__("moviepy"))
test("librosa",      lambda: __import__("librosa"))
test("speechrecog",  lambda: __import__("speech_recognition"))

print("\n=== Google AI ===")
test("google.genai", lambda: __import__("google.genai"))

print("\n=== App modules ===")
test("video_processor",    lambda: __import__("agent.video_processor", fromlist=["VideoProcessor"]))
test("facial_expression",  lambda: __import__("agent.analyzers.facial_expression", fromlist=["FacialExpressionAnalyzer"]))
test("body_language",      lambda: __import__("agent.analyzers.body_language", fromlist=["BodyLanguageAnalyzer"]))
test("voice_speech",       lambda: __import__("agent.analyzers.voice_speech", fromlist=["VoiceSpeechAnalyzer"]))
test("behavior_agent",     lambda: __import__("agent.behavior_agent", fromlist=["BehaviorAnalysisAgent"]))
test("gemini_counsellor",  lambda: __import__("agent.analyzers.gemini_counsellor", fromlist=["GeminiCounsellor"]))
test("context_intake",     lambda: __import__("agent.analyzers.context_intake", fromlist=["UserContext"]))
test("form_schema",        lambda: __import__("agent.twin.form_schema", fromlist=["TWIN_FORM_SCHEMA"]))

print("\n=== Summary ===")
failed = {k: v for k, v in results.items() if v != "OK"}
if not failed:
    print("  All OK — Flask should start fine.\n")
else:
    print(f"  {len(failed)} failures:")
    for k, v in failed.items():
        print(f"    {k}: {v}")

print()
