"""
Local SLM integration via Ollama.

  phi3       — interprets natural-language rules into structured config
  moondream  — analyzes a camera frame and describes what is happening
"""

import base64
import json
import threading
from typing import Optional

import cv2
import numpy as np
import requests

OLLAMA_BASE  = "http://localhost:11434"
TEXT_MODEL   = "phi3"
VISION_MODEL = "moondream"
TIMEOUT_TEXT   = 30
TIMEOUT_VISION = 45

_lock = threading.Lock()


def is_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _post(payload: dict, timeout: int) -> Optional[str]:
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/generate",
                          json=payload, timeout=timeout)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
    except Exception:
        pass
    return None


def describe_frame(frame: np.ndarray, event_type: str = "") -> str:
    """Send a camera frame to moondream; return a plain-English description."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(buf.tobytes()).decode()

    context = f'A security system flagged this frame for: "{event_type}". ' if event_type else ""
    prompt = (
        f"{context}Describe what is happening in this security camera image in 2-3 sentences. "
        "Focus on people's actions, any suspicious behaviour, vehicles, and objects. Be factual and concise."
    )

    with _lock:
        result = _post({"model": VISION_MODEL, "prompt": prompt,
                        "images": [b64], "stream": False}, TIMEOUT_VISION)
    return result or ""


def parse_rule(natural_language: str) -> dict:
    """
    Ask phi3 to convert a plain-English surveillance rule into structured JSON.
    Returns a dict with keys: objects, duration_seconds, alert_name, description.
    Returns {} on failure.
    """
    prompt = f"""You are a security camera rule parser. Convert the rule below to JSON.

Rule: "{natural_language}"

Output ONLY valid JSON with exactly these fields:
- "objects": list chosen from [person, car, truck, bus, motorcycle, bicycle, bottle, cup, chair, bench, dog, cat, bird, horse, sheep, cow, elephant, traffic light, fire hydrant, stop sign, parking meter, suitcase, backpack, umbrella]
- "duration_seconds": number — how long the object must be present before alerting (use 0 for instant)
- "alert_name": short UPPERCASE_UNDERSCORE string e.g. "ILLEGAL_PARKING"
- "description": one sentence explaining what this rule detects

Example:
{{"objects": ["car","truck"], "duration_seconds": 60, "alert_name": "ILLEGAL_PARKING", "description": "Vehicles parked in a restricted zone for over 1 minute."}}

JSON:"""

    with _lock:
        text = _post({"model": TEXT_MODEL, "prompt": prompt, "stream": False}, TIMEOUT_TEXT)

    if not text:
        return {}

    start = text.find("{")
    end   = text.rfind("}") + 1
    if start < 0 or end <= start:
        return {}

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}
