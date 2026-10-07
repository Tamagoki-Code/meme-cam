"""
Meme Cam
Make the right pose and the matching meme GIF (with sound) plays over your
webcam feed, positioned above whoever is doing the pose. Works with multiple
people in frame at once. Press q or Esc to quit, f to toggle fullscreen.

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
TARGET_FPS = 60           # requested without touching resolution (see open_camera)
MAX_PEOPLE = 4            # most people trackable at once
PERSON_MATCH_MAX_DIST = 0.3  # how close a hand/face must be to count as the same person
STICKY_FACE_SEC = 0.4     # keep using a person's last-seen face this long after a frame misses it
STICKY_HAND_SEC = 0.4     # same, for an individual hand (covering the face can hide it briefly)
DUPLICATE_HAND_MAX_DIST = 0.08  # two detections this close are treated as the same physical hand
WINDOW_SEC = 1.0          # how far back to measure hand movement
MIN_STEP = 0.004          # ignore tiny jitters between frames
KEEP_SEC = 1.0            # keep a GIF up this long after its trigger stops matching
GIF_WIDTH_FRACTION = 0.22  # fallback GIF width (as a fraction of frame width) when no face is found
GIF_TO_FACE_WIDTH_RATIO = 1.3  # GIF width relative to that person's face width
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
SALUTE_MAX_Y = 0.55       # wrist must be above this height (0 = top of frame)
SALUTE_MAX_MOTION = 0.15  # that hand's motion score must stay below this (held in place)
SALUTE_MAX_TEMPLE_DIST = 0.18  # that hand's wrist must stay within this of a temple landmark

# lebron scream: both hands on their matching temples (left hand-left temple, right-right)
LEBRON_TEMPLE_MAX_DIST = 0.18

# heart hands: thumbs touching, index fingers touching, forming a heart shape
HEART_MAX_FINGERTIP_DIST = 0.05

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


def draw_hand(frame, points, label):
    """points: 21 (x, y) tuples in normalized [0, 1] coordinates."""
    h, w = frame.shape[:2]
    color = HAND_COLORS.get(label, (180, 80, 255))
    pts = [(int(x * w), int(y * h)) for x, y in points]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(frame, p, 3, (255, 255, 255), -1)
    cv2.putText(frame, label, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


class HandSmoother:
    """Exponential moving average over each hand's 21 landmark points, to stop the
    overlay (and trigger geometry) from jittering frame to frame. Snaps instead of
    blending when a hand jumps too far, so fast motion isn't laggy/trailing."""

    def __init__(self, alpha=0.5, max_jump=0.2):
        self.alpha = alpha
        self.max_jump = max_jump
        self.last = {}  # label -> np.array(21, 2)

    def smooth(self, label, points):
        pts = np.array(points, dtype=np.float32)
        prev = self.last.get(label)
        if prev is None or np.linalg.norm(pts[0] - prev[0]) > self.max_jump:
            smoothed = pts
        else:
            smoothed = self.alpha * pts + (1 - self.alpha) * prev
        self.last[label] = smoothed
        return [tuple(p) for p in smoothed]


def index_finger_up(points):
    """True if the index finger is extended while the other three fingers are curled.
    points: 21 (x, y) tuples."""
    def extended(tip_idx, pip_idx):
        return points[tip_idx][1] < points[pip_idx][1]  # smaller y = higher up on screen
    return (extended(8, 6)
            and not extended(12, 10)
            and not extended(16, 14)
            and not extended(20, 18))


def build_face_data(fl):
    """Extracts the landmark points each trigger cares about, plus a face bounding box."""
    xs = [p.x for p in fl]
    ys = [p.y for p in fl]
    return {
        "nose": (fl[NOSE_TIP_IDX].x, fl[NOSE_TIP_IDX].y),
        "left_temple": (fl[LEFT_TEMPLE_IDX].x, fl[LEFT_TEMPLE_IDX].y),
        "right_temple": (fl[RIGHT_TEMPLE_IDX].x, fl[RIGHT_TEMPLE_IDX].y),
        "upper_lip": (fl[UPPER_LIP_IDX].x, fl[UPPER_LIP_IDX].y),
        "lower_lip": (fl[LOWER_LIP_IDX].x, fl[LOWER_LIP_IDX].y),
        "left_eye": (fl[LEFT_EYE_OUTER_IDX].x, fl[LEFT_EYE_OUTER_IDX].y),
        "right_eye": (fl[RIGHT_EYE_OUTER_IDX].x, fl[RIGHT_EYE_OUTER_IDX].y),
        "bbox": (min(xs), min(ys), max(xs), max(ys)),
    }


EMPTY_FACE = {"nose": None, "left_temple": None, "right_temple": None,
              "upper_lip": None, "lower_lip": None, "left_eye": None, "right_eye": None,
              "bbox": None}


def draw_face_points(frame, face):
    h, w = frame.shape[:2]
    for name, pt in face.items():
        if name == "bbox" or pt is None:
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
            frames.append(np.array(f.convert("RGBA")))
            durations.append(max(f.info.get("duration", 100), 40) / 1000)
        return frames, durations
    return [np.array(img.convert("RGBA"))], [1.0]


def frame_index(t, durations):
    for i, d in enumerate(durations):
        if t < d:
            return i
        t -= d
    return len(durations) - 1


def overlay(frame, rgba, x, y):
    """Composites rgba onto frame at (x, y), clipping to whatever part is on-screen
    (the sprite may be larger than the frame, or placed partly off its edges)."""
    h, w = rgba.shape[:2]
    H, W = frame.shape[:2]
    fx0, fy0 = max(x, 0), max(y, 0)
    fx1, fy1 = min(x + w, W), min(y + h, H)
    if fx0 >= fx1 or fy0 >= fy1:
        return
    sx0, sy0 = fx0 - x, fy0 - y
    sx1, sy1 = sx0 + (fx1 - fx0), sy0 + (fy1 - fy0)
    roi = frame[fy0:fy1, fx0:fx1]
    alpha = rgba[sy0:sy1, sx0:sx1, 3:4] / 255.0
    bgr = rgba[sy0:sy1, sx0:sx1, [2, 1, 0]]
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
    """Distance between two (x, y) points, or None if either is missing."""
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


## ---- meme triggers: each returns True/False from one person's landmarks this frame ----
# wrists: {"Left"/"Right": (x, y)}   fingers: {"Left"/"Right": {"thumb": (x,y), "index": (x,y)}}
# face: {"nose"/"left_temple"/"right_temple"/...: (x, y) or None}   tracker: that person's MotionTracker

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


# Order matters: the first matching trigger wins each frame, per person.
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
        frames=frames,          # original-resolution RGBA numpy arrays, one per GIF frame
        durations=durations,
        total=sum(durations),
        player=player,
        playing=False,
        last_trigger=-1e9,
        start=0.0,
        positions=[],           # where to draw it this frame: list of display_pos (see below)
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


def get_sprite(meme, now, target_w):
    t = (now - meme["start"]) % meme["total"]
    src = meme["frames"][frame_index(t, meme["durations"])]
    target_w = max(1, target_w)
    target_h = max(1, int(src.shape[0] * target_w / src.shape[1]))
    return cv2.resize(src, (target_w, target_h), interpolation=cv2.INTER_AREA)


def display_pos(person):
    """Where to draw a GIF for this person: above their face if we have one, else None."""
    bbox = person["face"].get("bbox")
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    return {"cx": (x0 + x1) / 2, "top": y0, "face_width": x1 - x0}


def draw_meme_at(meme, frame, now, pos):
    H, W = frame.shape[:2]
    if pos is None:
        target_w = int(W * GIF_WIDTH_FRACTION)
        sprite = get_sprite(meme, now, target_w)
        h, w = sprite.shape[:2]
        overlay(frame, sprite, W - w - 20, H - h - 20)
        return
    target_w = int(min(max(40, pos["face_width"] * W) * GIF_TO_FACE_WIDTH_RATIO, W * 1.2))
    sprite = get_sprite(meme, now, target_w)
    h, w = sprite.shape[:2]
    x = int(pos["cx"] * W - w / 2)
    y = int(pos["top"] * H - h - 10)  # place just above the top of the head
    overlay(frame, sprite, x, y)


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    # FPS only - deliberately not touching resolution, since changing it was what
    # seemed to be triggering the camera's onboard auto-framing crop.
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    return cap


class PeopleTracker:
    """
    Keeps a stable small-integer id per person across frames (by nearest face
    position), each with its own MotionTracker so one person's hand motion
    doesn't get mixed up with another's.

    Also keeps a short-lived "sticky" cache of each person's last-seen face and
    hands. Detection is noisy when a hand partially covers the face (e.g. the
    scuba pose) - without this, a single dropped frame would wipe out the hand
    or face and break a trigger that's genuinely still being held.
    """

    def __init__(self, max_people):
        self.max_people = max_people
        self.prev_centers = [None] * max_people
        self.prev_center_ts = [0.0] * max_people
        self.face_cache = [None] * max_people
        self.wrist_cache = [{} for _ in range(max_people)]  # label -> (wrist, fingers, ts)
        self.trackers = [MotionTracker() for _ in range(max_people)]

    def assign_faces(self, raw_faces, now):
        """raw_faces: list of face center (x, y) points. Returns {slot_id: raw_face_index}."""
        assigned = {}
        used_faces = set()
        for slot_id, prev_center in enumerate(self.prev_centers):
            if prev_center is None:
                continue
            best_idx, best_dist = None, PERSON_MATCH_MAX_DIST
            for idx, center in enumerate(raw_faces):
                if idx in used_faces:
                    continue
                d = dist(prev_center, center)
                if d is not None and d <= best_dist:
                    best_idx, best_dist = idx, d
            if best_idx is not None:
                assigned[slot_id] = best_idx
                used_faces.add(best_idx)
        free_slots = [s for s in range(self.max_people) if s not in assigned]
        for idx, center in enumerate(raw_faces):
            if idx in used_faces or not free_slots:
                continue
            slot_id = free_slots.pop(0)
            assigned[slot_id] = idx
            used_faces.add(idx)
        for slot_id in range(self.max_people):
            if slot_id in assigned:
                self.prev_centers[slot_id] = raw_faces[assigned[slot_id]]
                self.prev_center_ts[slot_id] = now
            elif (self.prev_centers[slot_id] is not None
                    and now - self.prev_center_ts[slot_id] > STICKY_FACE_SEC):
                self.prev_centers[slot_id] = None
                self.face_cache[slot_id] = None
        return assigned

    def recent_wrist(self, slot_id, label, now):
        cached = self.wrist_cache[slot_id].get(label)
        if cached is not None and now - cached[2] <= STICKY_HAND_SEC:
            return cached[0], cached[1]
        return None, None

    def remember_wrist(self, slot_id, label, wrist, fingers, now):
        self.wrist_cache[slot_id][label] = (wrist, fingers, now)

    def nearest_slot(self, people, point):
        best_slot, best_dist = None, None
        for slot_id, person in people.items():
            center = person["face"].get("nose") or self._bbox_center(person["face"].get("bbox"))
            d = dist(point, center)
            if d is not None and (best_dist is None or d < best_dist):
                best_slot, best_dist = slot_id, d
        return best_slot

    @staticmethod
    def _bbox_center(bbox):
        if bbox is None:
            return None
        x0, y0, x1, y1 = bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)


def main():
    for meme in MEMES:
        load_meme(meme)

    cap = open_camera()
    if not cap.isOpened():
        sys.exit("Could not open the camera. Try a different CAMERA_INDEX.")

    ensure_model()
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_PEOPLE * 2,
        min_hand_detection_confidence=0.4,
        min_tracking_confidence=0.3,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=MAX_PEOPLE,
        min_face_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    people_tracker = PeopleTracker(MAX_PEOPLE)
    hand_smoother = HandSmoother()
    t0, last_ts = time.time(), -1

    window_name = "Meme Cam"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
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

            # --- detect ---
            hand_res = hands.detect_for_video(image, ts)
            candidates = []
            for lm, hd in zip(hand_res.hand_landmarks, hand_res.handedness):
                # frame is mirrored for display, so flip the reported handedness back
                label = "Right" if hd[0].category_name == "Left" else "Left"
                candidates.append({"lm": lm, "label": label, "score": hd[0].score,
                                    "wrist_raw": (lm[0].x, lm[0].y)})

            # the detector occasionally reports the same physical hand twice (often
            # misclassifying it as both Left and Right) - drop the lower-confidence copy
            candidates.sort(key=lambda c: c["score"], reverse=True)
            kept = []
            for c in candidates:
                if not any(dist(c["wrist_raw"], k["wrist_raw"]) <= DUPLICATE_HAND_MAX_DIST
                           for k in kept):
                    kept.append(c)

            raw_hands = []
            for c in kept:
                lm, label = c["lm"], c["label"]
                points = hand_smoother.smooth(label, [(p.x, p.y) for p in lm])
                raw_hands.append({
                    "label": label,
                    "wrist": points[0],
                    "wrist_raw": (lm[0].x, lm[0].y),  # unsmoothed, for measuring real motion
                    "fingers": {
                        "thumb": points[4],
                        "index": points[8],
                        "index_up": index_finger_up(points),
                    },
                    "points": points,
                })

            face_res = face_landmarker.detect_for_video(image, ts)
            raw_faces = [build_face_data(fl) for fl in face_res.face_landmarks]
            raw_face_centers = [f["nose"] for f in raw_faces]

            # --- group detections into people, keeping ids stable across frames ---
            assigned = people_tracker.assign_faces(raw_face_centers, now)
            people = {}
            for slot_id, face_idx in assigned.items():
                people_tracker.face_cache[slot_id] = raw_faces[face_idx]
                people[slot_id] = {"face": raw_faces[face_idx], "wrists": {}, "fingers": {}}
            for slot_id in range(MAX_PEOPLE):
                # a face briefly lost to occlusion (e.g. a hand covering it) still counts
                # as present for a moment, using its last-known landmarks
                if (slot_id not in people and people_tracker.prev_centers[slot_id] is not None
                        and people_tracker.face_cache[slot_id] is not None):
                    people[slot_id] = {"face": people_tracker.face_cache[slot_id],
                                        "wrists": {}, "fingers": {}, "wrists_raw": {}}
            if not people:
                # nobody's face was detected; still let hand-only poses work for one person
                people[0] = {"face": dict(EMPTY_FACE), "wrists": {}, "fingers": {}, "wrists_raw": {}}
            for person in people.values():
                person.setdefault("wrists_raw", {})

            for raw_hand in raw_hands:
                slot_id = people_tracker.nearest_slot(people, raw_hand["wrist"])
                if slot_id is None:
                    slot_id = next(iter(people))
                people[slot_id]["wrists"][raw_hand["label"]] = raw_hand["wrist"]
                people[slot_id]["fingers"][raw_hand["label"]] = raw_hand["fingers"]
                people[slot_id]["wrists_raw"][raw_hand["label"]] = raw_hand["wrist_raw"]
                people_tracker.remember_wrist(
                    slot_id, raw_hand["label"], raw_hand["wrist"], raw_hand["fingers"], now)
                if SHOW_LANDMARKS:
                    draw_hand(frame, raw_hand["points"], raw_hand["label"])

            # a hand briefly lost (e.g. occluded while covering the face) stays "present"
            # for a moment at its last-known spot, instead of instantly dropping the pose
            for slot_id, person in people.items():
                for label in ("Left", "Right"):
                    if label in person["wrists"]:
                        continue
                    wrist, fingers = people_tracker.recent_wrist(slot_id, label, now)
                    if wrist is not None:
                        person["wrists"][label] = wrist
                        person["fingers"][label] = fingers

            if SHOW_LANDMARKS:
                for person in people.values():
                    if person["face"].get("nose") is not None:
                        draw_face_points(frame, person["face"])

            # --- per person, find the first matching meme this frame ---
            meme_positions = {meme["name"]: [] for meme in MEMES}
            matches_debug = []
            for slot_id, person in people.items():
                tracker = people_tracker.trackers[slot_id]
                tracker.update(now, person["wrists_raw"])
                for meme in MEMES:
                    if meme["trigger"](person["wrists"], tracker, person["face"], person["fingers"]):
                        meme_positions[meme["name"]].append(display_pos(person))
                        matches_debug.append(meme["name"])
                        break

            # --- update + draw each meme that's currently showing ---
            for meme in MEMES:
                positions = meme_positions[meme["name"]]
                active = len(positions) > 0
                showing = update_meme(meme, now, active)
                if active:
                    meme["positions"] = positions
                if showing:
                    for pos in meme["positions"]:
                        draw_meme_at(meme, frame, now, pos)

            for meme in MEMES:
                if meme["player"] is not None:
                    pyglet.clock.tick()
                    pyglet.app.platform_event_loop.step(0)
                    break

            if SHOW_DEBUG:
                cv2.putText(frame,
                            f"people: {len(people)}  match: {', '.join(matches_debug) or '-'}",
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
