"""
Meme Cam
Make the right pose and the matching meme GIF (with sound) plays over your
webcam feed. Press q or Esc to quit, f to toggle fullscreen.

Setup:
    pip install opencv-python mediapipe pillow numpy pyglet
    python scuba_cat.py

To add a new meme:
    1. Drop gifs/<name>.<gif|png|jpg|...> and mp3s/<name>.mp3 (mp3 optional).
    2. Write a trigger(wrists, tracker, face, fingers) -> bool function using
       the near() / near_any() / hands_raised_and_level() / hand_motion()
       helpers above the trigger functions (see the existing ones for examples).
    3. Add {"name": "<name>", "trigger": your_trigger} to MEMES, ordered so
       more specific poses come before poses they could be mistaken for.
"""
import os
import sys
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyglet
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageSequence

# ---------------- settings you can tweak ----------------
CAMERA_INDEX = 0
WINDOW_SEC = 1.0          # how far back to measure hand movement
MIN_STEP = 0.004          # ignore tiny jitters between frames
KEEP_SEC = 1.0            # keep a GIF up this long after its trigger stops matching
GIF_WIDTH_FRACTION = 0.4  # GIF width as a fraction of the camera frame
SHOW_LANDMARKS = True
SHOW_DEBUG = True

# scuba cat: keep left hand near the nose, paddle with the right hand
SCUBA_MOTION_THRESHOLD = 0.35
SCUBA_PADDLE_HAND = "Right"
SCUBA_NOSE_HAND = "Left"
SCUBA_NOSE_HAND_MAX_DIST = 0.25  # nose hand wrist must stay within this of the nose landmark

# absolute cinema: both hands raised and held level with each other, away from the temples
CINEMA_MAX_Y = 0.55       # wrists must be above this height (0 = top of frame)
CINEMA_LEVEL_TOL = 0.08   # max allowed height difference between the two wrists
CINEMA_MIN_TEMPLE_DIST = 0.22  # each wrist must be at least this far from either temple

# salute: one hand raised up near the head/temple and held mostly still
SALUTE_MAX_Y = 0.35       # wrist must be above this height (0 = top of frame)
SALUTE_MAX_MOTION = 0.15  # that hand's motion score must stay below this (held in place)
SALUTE_MAX_TEMPLE_DIST = 0.18  # that hand's wrist must stay within this of a temple landmark

# lebron scream: both hands on their matching temples (left hand-left temple, right-right)
LEBRON_TEMPLE_MAX_DIST = 0.18

# heart hands: thumbs touching, index fingers touching, forming a heart shape
HEART_MAX_FINGERTIP_DIST = 0.09

# emoji nerd: mouth open and one index finger pointed up (like adjusting glasses)
NERD_MOUTH_OPEN_RATIO = 0.3  # mouth-gap / eye-distance must be at least this

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

FACE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                   "face_landmarker/float16/1/face_landmarker.task")
# indices into the 478-point MediaPipe face mesh
NOSE_TIP_IDX = 1
LEFT_TEMPLE_IDX = 127
RIGHT_TEMPLE_IDX = 356
UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14
LEFT_EYE_OUTER_IDX = 263
RIGHT_EYE_OUTER_IDX = 33


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand tracking model (one time, ~8 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    if not os.path.exists(FACE_MODEL_PATH):
        print("Downloading face tracking model (one time, ~4 MB)...")
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)


HAND_COLORS = {"Left": (60, 60, 255), "Right": (60, 220, 60)}  # BGR: Left=red, Right=green


def draw_hand(frame, landmarks, label):
    h, w = frame.shape[:2]
    color = HAND_COLORS.get(label, (180, 80, 255))
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(frame, p, 3, (255, 255, 255), -1)
    cv2.putText(frame, label, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def index_finger_up(lm):
    """True if the index finger is extended while the other three fingers are curled."""
    def extended(tip_idx, pip_idx):
        return lm[tip_idx].y < lm[pip_idx].y  # smaller y = higher up on screen
    return (extended(8, 6)
            and not extended(12, 10)
            and not extended(16, 14)
            and not extended(20, 18))


def draw_face_points(frame, face):
    h, w = frame.shape[:2]
    for name, pt in face.items():
        if pt is None:
            continue
        cv2.circle(frame, (int(pt[0] * w), int(pt[1] * h)), 4, (0, 255, 255), -1)
        cv2.putText(frame, name, (int(pt[0] * w) + 6, int(pt[1] * h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)


def load_image_frames(path):
    """Loads a GIF as (frames, durations), or a still image as a single frame."""
    img = Image.open(path)
    if getattr(img, "is_animated", False):
        frames, durations = [], []
        for f in ImageSequence.Iterator(img):
            frames.append(f.convert("RGBA"))
            durations.append(max(f.info.get("duration", 100), 40) / 1000)
        return frames, durations
    return [img.convert("RGBA")], [1.0]


def resize_frames(frames, target_w):
    out = []
    for f in frames:
        h = int(f.height * target_w / f.width)
        out.append(np.array(f.resize((target_w, h), Image.LANCZOS)))
    return out


def frame_index(t, durations):
    for i, d in enumerate(durations):
        if t < d:
            return i
        t -= d
    return len(durations) - 1


def overlay(frame, rgba, x, y):
    h, w = rgba.shape[:2]
    H, W = frame.shape[:2]
    x, y = max(0, min(x, W - w)), max(0, min(y, H - h))
    roi = frame[y:y + h, x:x + w]
    alpha = rgba[:, :, 3:4] / 255.0
    bgr = rgba[:, :, [2, 1, 0]]
    roi[:] = (alpha * bgr + (1 - alpha) * roi).astype(np.uint8)


class MotionTracker:
    """Adds up how far each wrist has travelled over the last WINDOW_SEC."""

    def __init__(self):
        self.last = {}
        self.steps = deque()  # (time, hand label, distance moved)

    def update(self, t, wrists):
        for label, (x, y) in wrists.items():
            if label in self.last:
                px, py = self.last[label]
                step = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                if step >= MIN_STEP:
                    self.steps.append((t, label, step))
            self.last[label] = (x, y)
        for label in list(self.last):
            if label not in wrists:
                del self.last[label]
        while self.steps and t - self.steps[0][0] > WINDOW_SEC:
            self.steps.popleft()

    def scores(self):
        s = {"Left": 0.0, "Right": 0.0}
        for _, label, step in self.steps:
            s[label] += step
        return s


## ---- reusable geometry helpers for writing new triggers ----

def dist(a, b):
    """Distance between two (x, y) points, or False-y if either is missing."""
    if a is None or b is None:
        return None
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def near(a, b, max_dist):
    """True if a and b are both present and within max_dist of each other."""
    d = dist(a, b)
    return d is not None and d <= max_dist


def near_any(point, candidates, max_dist):
    """True if point is near at least one of candidates (None entries ignored)."""
    return any(near(point, c, max_dist) for c in candidates if c is not None)


def both_hands_near_matching(wrists, face, left_key, right_key, max_dist):
    """True if the left wrist is near face[left_key] and the right wrist near face[right_key]."""
    if "Left" not in wrists or "Right" not in wrists:
        return False
    return (near(wrists["Left"], face.get(left_key), max_dist)
            and near(wrists["Right"], face.get(right_key), max_dist))


def hands_raised_and_level(wrists, max_y, level_tol):
    """True if both wrists are above max_y (smaller y = higher) and level with each other."""
    if "Left" not in wrists or "Right" not in wrists:
        return False
    ly, ry = wrists["Left"][1], wrists["Right"][1]
    return ly <= max_y and ry <= max_y and abs(ly - ry) <= level_tol


def hand_motion(tracker, label):
    return tracker.scores()[label]


## ---- meme triggers: each returns True/False from this frame's landmarks ----
# wrists: {"Left"/"Right": (x, y)}   fingers: {"Left"/"Right": {"thumb": (x,y), "index": (x,y)}}
# face: {"nose"/"left_temple"/"right_temple": (x, y) or None}   tracker: MotionTracker

def scuba_cat_trigger(wrists, tracker, face, fingers):
    paddling = hand_motion(tracker, SCUBA_PADDLE_HAND) >= SCUBA_MOTION_THRESHOLD
    if not paddling or SCUBA_NOSE_HAND not in wrists:
        return False
    if face.get("nose") is None:
        return True  # no face detected, fall back to requiring the nose hand to just be present
    return near(wrists[SCUBA_NOSE_HAND], face["nose"], SCUBA_NOSE_HAND_MAX_DIST)


def absolute_cinema_trigger(wrists, tracker, face, fingers):
    if not hands_raised_and_level(wrists, CINEMA_MAX_Y, CINEMA_LEVEL_TOL):
        return False
    temples = [p for p in (face.get("left_temple"), face.get("right_temple")) if p is not None]
    if not temples:
        return True  # no face detected, fall back to the raised+level check
    return all(not near_any(w, temples, CINEMA_MIN_TEMPLE_DIST) for w in wrists.values())


def salute_trigger(wrists, tracker, face, fingers):
    temples = [p for p in (face.get("left_temple"), face.get("right_temple")) if p is not None]
    for label, wrist in wrists.items():
        if wrist[1] > SALUTE_MAX_Y or hand_motion(tracker, label) > SALUTE_MAX_MOTION:
            continue
        if not temples:
            return True  # no face detected, fall back to the height+stillness check
        if near_any(wrist, temples, SALUTE_MAX_TEMPLE_DIST):
            return True
    return False


def lebron_scream_trigger(wrists, tracker, face, fingers):
    return both_hands_near_matching(
        wrists, face, "left_temple", "right_temple", LEBRON_TEMPLE_MAX_DIST)


def heart_hands_trigger(wrists, tracker, face, fingers):
    if "Left" not in fingers or "Right" not in fingers:
        return False
    return (near(fingers["Left"]["thumb"], fingers["Right"]["thumb"], HEART_MAX_FINGERTIP_DIST)
            and near(fingers["Left"]["index"], fingers["Right"]["index"], HEART_MAX_FINGERTIP_DIST))


def emoji_nerd_trigger(wrists, tracker, face, fingers):
    pointer_up = any(fingers.get(label, {}).get("index_up") for label in ("Left", "Right"))
    if not pointer_up:
        return False
    mouth_gap = dist(face.get("upper_lip"), face.get("lower_lip"))
    eye_dist = dist(face.get("left_eye"), face.get("right_eye"))
    if mouth_gap is None or not eye_dist:
        return False
    return (mouth_gap / eye_dist) >= NERD_MOUTH_OPEN_RATIO


# Order matters: the first matching trigger wins each frame.
MEMES = [
    {"name": "absolute_cinema", "trigger": absolute_cinema_trigger},
    {"name": "lebron_scream", "trigger": lebron_scream_trigger},
    {"name": "heart_hands", "trigger": heart_hands_trigger},
    {"name": "emoji_nerd", "trigger": emoji_nerd_trigger},
    {"name": "salute", "trigger": salute_trigger},
    {"name": "scuba_cat", "trigger": scuba_cat_trigger},
]


def find_image_path(meme_name):
    for ext in (".gif", ".png", ".jpg", ".jpeg", ".webp"):
        path = os.path.join("gifs", f"{meme_name}{ext}")
        if os.path.exists(path):
            return path
    return os.path.join("gifs", f"{meme_name}.gif")  # default, for the error message


def load_meme(meme):
    gif_path = find_image_path(meme["name"])
    mp3_path = os.path.join("mp3s", f"{meme['name']}.mp3")
    try:
        frames, durations = load_image_frames(gif_path)
    except FileNotFoundError:
        sys.exit(f"Could not find an image for {meme['name']} in gifs/.")
    player = None
    if os.path.exists(mp3_path):
        player = pyglet.media.Player()
        player.queue(pyglet.media.load(mp3_path, streaming=False))
        player.loop = True
    else:
        print(f"No sound found at {mp3_path}, running without audio for {meme['name']}.")
    meme.update(
        frames=frames,
        durations=durations,
        total=sum(durations),
        sprites=None,
        player=player,
        playing=False,
        last_trigger=-1e9,
        start=0.0,
    )


def update_meme(meme, now, active):
    if active:
        if now - meme["last_trigger"] > KEEP_SEC:  # wasn't showing yet: restart the GIF
            meme["start"] = now
        meme["last_trigger"] = now
    showing = now - meme["last_trigger"] <= KEEP_SEC

    player = meme["player"]
    if player is not None:
        if showing and not meme["playing"]:
            player.seek(0)
            player.play()
            meme["playing"] = True
        elif not showing and meme["playing"]:
            player.pause()
            meme["playing"] = False

    return showing


def draw_meme(meme, frame, now):
    if meme["sprites"] is None:
        meme["sprites"] = resize_frames(meme["frames"], int(frame.shape[1] * GIF_WIDTH_FRACTION))
    t = (now - meme["start"]) % meme["total"]
    sprite = meme["sprites"][frame_index(t, meme["durations"])]
    h, w = sprite.shape[:2]
    overlay(frame, sprite, frame.shape[1] - w - 20, frame.shape[0] - h - 20)


def main():
    for meme in MEMES:
        load_meme(meme)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit("Could not open the camera. Try a different CAMERA_INDEX.")

    ensure_model()
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    tracker = MotionTracker()
    t0, last_ts = time.time(), -1

    window_name = "Meme Cam"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    fullscreen = False

    with vision.HandLandmarker.create_from_options(hand_options) as hands, \
            vision.FaceLandmarker.create_from_options(face_options) as face_landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror view
            now = time.time()

            ts = max(int((now - t0) * 1000), last_ts + 1)  # must keep increasing
            last_ts = ts
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            res = hands.detect_for_video(image, ts)
            wrists = {}
            fingers = {}
            for lm, hd in zip(res.hand_landmarks, res.handedness):
                # frame is mirrored for display, so flip the reported handedness back
                label = "Right" if hd[0].category_name == "Left" else "Left"
                wrists[label] = (lm[0].x, lm[0].y)
                fingers[label] = {
                    "thumb": (lm[4].x, lm[4].y),
                    "index": (lm[8].x, lm[8].y),
                    "index_up": index_finger_up(lm),
                }
                if SHOW_LANDMARKS:
                    draw_hand(frame, lm, label)

            face_res = face_landmarker.detect_for_video(image, ts)
            face = {"nose": None, "left_temple": None, "right_temple": None,
                    "upper_lip": None, "lower_lip": None, "left_eye": None, "right_eye": None}
            if face_res.face_landmarks:
                fl = face_res.face_landmarks[0]
                face["nose"] = (fl[NOSE_TIP_IDX].x, fl[NOSE_TIP_IDX].y)
                face["left_temple"] = (fl[LEFT_TEMPLE_IDX].x, fl[LEFT_TEMPLE_IDX].y)
                face["right_temple"] = (fl[RIGHT_TEMPLE_IDX].x, fl[RIGHT_TEMPLE_IDX].y)
                face["upper_lip"] = (fl[UPPER_LIP_IDX].x, fl[UPPER_LIP_IDX].y)
                face["lower_lip"] = (fl[LOWER_LIP_IDX].x, fl[LOWER_LIP_IDX].y)
                face["left_eye"] = (fl[LEFT_EYE_OUTER_IDX].x, fl[LEFT_EYE_OUTER_IDX].y)
                face["right_eye"] = (fl[RIGHT_EYE_OUTER_IDX].x, fl[RIGHT_EYE_OUTER_IDX].y)
                if SHOW_LANDMARKS:
                    draw_face_points(frame, face)

            tracker.update(now, wrists)

            matched = None
            for meme in MEMES:
                if matched is None and meme["trigger"](wrists, tracker, face, fingers):
                    matched = meme["name"]

            active_meme = None
            for meme in MEMES:
                showing = update_meme(meme, now, active=(meme["name"] == matched))
                if showing:
                    active_meme = meme

            if active_meme is not None:
                draw_meme(active_meme, frame, now)

            for meme in MEMES:
                if meme["player"] is not None:
                    pyglet.clock.tick()
                    pyglet.app.platform_event_loop.step(0)
                    break

            if SHOW_DEBUG:
                s = tracker.scores()
                cv2.putText(frame,
                            f"match: {matched or '-'}  "
                            f"L {s['Left']:.2f} R {s['Right']:.2f}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    window_name, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()
    for meme in MEMES:
        if meme["player"] is not None:
            meme["player"].pause()


if __name__ == "__main__":
    main()
