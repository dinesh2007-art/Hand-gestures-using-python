"""
Real-Time Hand Gesture Recognition with OpenCV + MediaPipe (Python 3.10.9)

Gestures recognized (simple heuristics):
- OPEN_PALM        : all five fingers extended
- FIST             : all fingers curled
- PEACE            : index + middle extended, others curled
- THUMBS_UP        : thumb extended, other fingers curled
- OK_SIGN          : thumb & index touching (circle), other fingers mostly curled
Behavior:
- Reads frames from default webcam (index 0), mirrored for a "selfie" preview
- Uses MediaPipe Hands to get landmarks
- Simple rules to decide which gesture is shown
- Debounces detections (require N consecutive frames)
- On stable detection: prints gesture and performs a placeholder API POST
- Loads API key from environment variable: GESTURE_API_KEY

Setup (Terminal):
    python -m venv .venv
    .venv\Scripts\activate            # Windows
    # source .venv/bin/activate       # macOS / Linux
    pip install opencv-python mediapipe requests python-dotenv streamlit

Environment (optional .env file for local dev):
    GESTURE_API_KEY=YOUR_API_KEY

Run:
    streamlit run gesture.py

VS Code tips:
- Use the Python extension, select your venv interpreter
- Add a launch.json entry with "console": "integratedTerminal"
- Set breakpoints in the helper functions (finger state, gesture rules)
"""

from __future__ import annotations

import os
import time
import math
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Third-party
import cv2
import requests
import streamlit as st

# MediaPipe
import mediapipe as mp

# Optional: load environment variables from .env if present (dev convenience)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# --------------------------- Configuration -----------------------------------

API_ENDPOINT = "https://api.groq.com/gesture_action"  # <-- replace if needed
API_KEY_ENV = "GESTURE_API_KEY"  # env var name for your key
DEFAULT_COOLDOWN_SEC = 1.0       # time to wait before allowing next API call for same gesture
STABLE_FRAMES_REQUIRED = 5       # how many consecutive frames required to confirm a gesture
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
MAX_NUM_HANDS = 1                # demo targets single hand for simpler logic

# Logging for easy debugging in VS Code
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("gesture-app")

# --------------------------- Data Structures ---------------------------------

@dataclass
class GestureEvent:
    name: str
    confidence: float  # heuristic confidence in [0,1]
    timestamp: float


# --------------------------- Utility Functions --------------------------------

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two 2D points (normalized coords)."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def norm(v: Tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def unit(v: Tuple[float, float]) -> Tuple[float, float]:
    n = norm(v)
    if n == 0:
        return (0.0, 0.0)
    return (v[0] / n, v[1] / n)


def dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


# --------------------------- Finger State Logic --------------------------------

# MediaPipe landmark indices (for readability)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


def fingers_up(
    lm: List[Tuple[float, float]],
    handedness_label: str,
) -> Dict[str, bool]:
    """
    Determine which fingers are extended using simple rules on normalized coords.
    Assumes the image is mirrored (cv2.flip(..., 1)) *before* processing by MediaPipe—
    which matches the "selfie" preview and is common in apps.

    - For Index/Middle/Ring/Pinky: finger is "up" if tip_y < pip_y (lower y = higher on screen).
    - For Thumb: depends on hand:
        * 'Right' hand: thumb is "up/extended" if tip_x > ip_x (points to image-right)
        * 'Left'  hand: thumb is "up/extended" if tip_x < ip_x (points to image-left)

    Returns a dict like {"thumb": True/False, "index": ..., ...}
    """
    # Other fingers: check vertical extension
    index_up = lm[INDEX_TIP][1] < lm[INDEX_PIP][1]
    middle_up = lm[MIDDLE_TIP][1] < lm[MIDDLE_PIP][1]
    ring_up = lm[RING_TIP][1] < lm[RING_PIP][1]
    pinky_up = lm[PINKY_TIP][1] < lm[PINKY_PIP][1]

    # Thumb: check horizontal, using handedness
    if handedness_label == "Right":
        thumb_up = lm[THUMB_TIP][0] > lm[THUMB_IP][0]
    else:
        thumb_up = lm[THUMB_TIP][0] < lm[THUMB_IP][0]

    return {
        "thumb": bool(thumb_up),
        "index": bool(index_up),
        "middle": bool(middle_up),
        "ring": bool(ring_up),
        "pinky": bool(pinky_up),
    }


def is_ok_sign(lm: List[Tuple[float, float]]) -> Tuple[bool, float]:
    """
    Heuristic OK sign: thumb_tip and index_tip close together relative to hand size.
    Confidence is inversely proportional to that distance.

    We normalize by the distance between wrist and middle_mcp to account for scale.
    """
    wrist = (lm[WRIST][0], lm[WRIST][1])
    mid_mcp = (lm[MIDDLE_MCP][0], lm[MIDDLE_MCP][1])
    ref = max(distance(wrist, mid_mcp), 1e-6)

    d = distance((lm[THUMB_TIP][0], lm[THUMB_TIP][1]),
                 (lm[INDEX_TIP][0], lm[INDEX_TIP][1]))
    ratio = d / ref
    ok = ratio < 0.35  # threshold from quick experimentation; tweak in practice
    conf = float(max(0.0, min(1.0, 1.0 - ratio)))
    return ok, conf


def classify_gesture(lm: List[Tuple[float, float]], handedness: str) -> GestureEvent:
    """
    Produce a GestureEvent with name and a rough confidence score.
    Prioritized rules (first match wins).
    """
    f = fingers_up(lm, handedness)

    # OPEN_PALM
    if all(f.values()):
        return GestureEvent("OPEN_PALM", confidence=0.9, timestamp=time.time())

    # FIST
    if not any(f.values()):
        return GestureEvent("FIST", confidence=0.9, timestamp=time.time())

    # PEACE (index + middle up, others down)
    if f["index"] and f["middle"] and not (f["ring"] or f["pinky"] or f["thumb"]):
        return GestureEvent("PEACE", confidence=0.85, timestamp=time.time())

    # THUMBS_UP (thumb up, others down)
    if f["thumb"] and not (f["index"] or f["middle"] or f["ring"] or f["pinky"]):
        return GestureEvent("THUMBS_UP", confidence=0.85, timestamp=time.time())

    # OK_SIGN (circle thumb-index) + not many other fingers up
    ok, ok_conf = is_ok_sign(lm)
    if ok and sum([f["middle"], f["ring"], f["pinky"]]) <= 1:
        return GestureEvent("OK_SIGN", confidence=max(0.75, ok_conf), timestamp=time.time())

    # Fallback: unknown gesture (low confidence)
    return GestureEvent("UNKNOWN", confidence=0.0, timestamp=time.time())


# --------------------------- API Integration (Placeholder) --------------------

def get_api_key() -> Optional[str]:
    """
    Return the API key: prefers env var GESTURE_API_KEY, otherwise checks placeholder.
    """
    key = os.getenv(API_KEY_ENV)
    if key and key.strip() and key.strip() != "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":
        return key.strip()
    return None


def send_gesture_api_call(gesture: GestureEvent) -> None:
    """
    Placeholder API call to demonstrate integration.
    - Uses Bearer token from env var
    - Sends JSON payload with gesture info
    """
    key = get_api_key()
    if not key:
        logger.warning(
            "API key missing. Set %s env var (e.g., in .env) to enable API calls.",
            API_KEY_ENV,
        )
        return

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "gesture": gesture.name,
        "confidence": gesture.confidence,
        "timestamp": gesture.timestamp,
        "source": "opencv-mediapipe-demo",
    }

    try:
        resp = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=5)
        logger.info("API response: %s %s", resp.status_code, resp.text[:200])
    except requests.RequestException as e:
        logger.error("API request failed: %s", e)


# --------------------------- Main Application ---------------------------------

def main() -> None:
    st.set_page_config(page_title="Gesture Recognition", layout="wide")
    st.title("Real-Time Hand Gesture Recognition")
    st.markdown("Use your webcam to detect hand gestures in real-time.")

    run = st.checkbox("Run Webcam")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        frame_window = st.image([])
        
    with col2:
        st.markdown("### Status")
        gesture_text = st.empty()
        confidence_text = st.empty()
        api_status = st.empty()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    if run:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # On Linux/macOS, omit CAP_DSHOW
        if not cap.isOpened():
            st.error("Unable to open webcam. Check your camera or index.")
            return

        logger.info("Starting camera.")

        # Gesture debouncing state
        last_label: Optional[str] = None
        stable_count = 0
        last_api_time: Dict[str, float] = {}

        with mp_hands.Hands(
            model_complexity=1,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        ) as hands:

            while run:
                ok, frame = cap.read()
                if not ok:
                    st.warning("Frame grab failed; retrying...")
                    time.sleep(0.1)
                    continue

                # Mirror image for selfie view and pass to MediaPipe
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = hands.process(frame_rgb)

                gesture_label_to_show = "No hand"
                confidence_to_show = 0.0

                if results.multi_hand_landmarks and results.multi_handedness:
                    # Only consider the first detected hand (as configured)
                    hand_landmarks = results.multi_hand_landmarks[0]
                    handedness = results.multi_handedness[0].classification[0].label  # 'Left'/'Right'

                    # Convert landmarks to normalized (x,y) tuples for convenience
                    lm_xy: List[Tuple[float, float]] = [
                        (lm.x, lm.y) for lm in hand_landmarks.landmark
                    ]

                    # Classify gesture
                    gesture = classify_gesture(lm_xy, handedness)
                    gesture_label_to_show = gesture.name
                    confidence_to_show = gesture.confidence

                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    # Debounce: require same label for N frames
                    if gesture.name == last_label and gesture.name != "UNKNOWN":
                        stable_count += 1
                    else:
                        stable_count = 1
                        last_label = gesture.name

                    # If stable, print & call API (cooldown per gesture)
                    if stable_count >= STABLE_FRAMES_REQUIRED and gesture.name != "UNKNOWN":
                        now = time.time()
                        last_time = last_api_time.get(gesture.name, 0.0)
                        if now - last_time >= DEFAULT_COOLDOWN_SEC:
                            logger.info(f"[GESTURE] {gesture.name} (conf={gesture.confidence:.2f})")
                            api_status.success(f"**{gesture.name}** detected! API called at {time.strftime('%X')}")
                            send_gesture_api_call(gesture)
                            last_api_time[gesture.name] = now

                # HUD overlay
                cv2.rectangle(frame, (10, 10), (310, 90), (0, 0, 0), -1)
                cv2.putText(frame, f"Gesture: {gesture_label_to_show}",
                            (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Conf: {confidence_to_show:.2f}",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)

                # Update Streamlit UI
                gesture_text.markdown(f"**Current Gesture:** {gesture_label_to_show}")
                confidence_text.markdown(f"**Confidence:** {confidence_to_show:.2f}")

                # Convert BGR back to RGB for displaying in Streamlit
                frame_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_out)

        cap.release()
        logger.info("Stopped.")
    else:
        st.info("Click 'Run Webcam' to start gesture recognition.")


if __name__ == "__main__":
    main()
