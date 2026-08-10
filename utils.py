"""Shared helpers for Cartman Pose App (MediaPipe + OpenCV)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
DATA = ROOT / "data"
MODELS = ROOT / "models"

# Class → sticker (Prediction N → assets/N.*)
STICKER_FILES = {
    0: "0.jpg",  # scheming / begging Cartman
    1: "1.gif",  # crying Cartman
    2: "2.png",  # buddha Cartman
    3: "3.jpg",  # anime cheek-cup meme
    4: "4.jpg",  # crossed-arms Cartman
    5: "5.png",  # pointer-fingers-up Cartman
}

CLASS_NAMES = {
    0: "begging",
    1: "crying",
    2: "buddha",
    3: "anime",
    4: "crossed",
    5: "pointers",
}

NUM_HAND_LANDMARKS = 21
FEATURES_PER_HAND = NUM_HAND_LANDMARKS * 2  # x,y wrist-relative
FACE_FEATURE_COUNT = 6  # mouth, L/R ear, nose-normalized extras
FEATURE_DIM = FEATURES_PER_HAND * 2 + FACE_FEATURE_COUNT  # 2 hands + face
NUM_CLASSES = 6

# Face mesh landmark indices
LIP_UPPER, LIP_LOWER = 13, 14
# Eye aspect: upper / lower lid midpoints
LEFT_EYE_UP, LEFT_EYE_LOW = 159, 145
RIGHT_EYE_UP, RIGHT_EYE_LOW = 386, 374
LEFT_EYE_OUTER, LEFT_EYE_INNER = 33, 133
RIGHT_EYE_OUTER, RIGHT_EYE_INNER = 362, 263
# Cheeks / face sides
LEFT_CHEEK, RIGHT_CHEEK = 234, 454
CHIN = 152
NOSE = 1


def ensure_dirs() -> None:
    ASSETS.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)


def landmarks_to_row(hand_landmarks) -> list[float]:
    pts = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    base_x, base_y = pts[0]
    row: list[float] = []
    for x, y in pts:
        row.append(x - base_x)
        row.append(y - base_y)
    return row


def pad_features(row: list[float], width: int = FEATURES_PER_HAND) -> list[float]:
    if len(row) >= width:
        return row[:width]
    return row + [0.0] * (width - len(row))


def _dist(a, b) -> float:
    return float(((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5)


def face_feature_vector(face_landmarks) -> list[float]:
    """Compact face cues for crying / cheek-cup gestures."""
    if face_landmarks is None:
        return [0.0] * FACE_FEATURE_COUNT

    face_w = _dist(
        face_landmarks.landmark[LEFT_EYE_OUTER],
        face_landmarks.landmark[RIGHT_EYE_OUTER],
    )
    face_w = max(face_w, 1e-6)
    mouth = _dist(face_landmarks.landmark[LIP_UPPER], face_landmarks.landmark[LIP_LOWER]) / face_w

    def ear(up, low, outer, inner) -> float:
        vert = _dist(face_landmarks.landmark[up], face_landmarks.landmark[low])
        horiz = _dist(face_landmarks.landmark[outer], face_landmarks.landmark[inner])
        return vert / max(horiz, 1e-6)

    left_ear = ear(LEFT_EYE_UP, LEFT_EYE_LOW, LEFT_EYE_OUTER, LEFT_EYE_INNER)
    right_ear = ear(RIGHT_EYE_UP, RIGHT_EYE_LOW, RIGHT_EYE_OUTER, RIGHT_EYE_INNER)
    nose = face_landmarks.landmark[NOSE]
    chin = face_landmarks.landmark[CHIN]
    left_cheek = face_landmarks.landmark[LEFT_CHEEK]
    right_cheek = face_landmarks.landmark[RIGHT_CHEEK]
    return [
        mouth,
        left_ear,
        right_ear,
        chin.y - nose.y,
        left_cheek.x - nose.x,
        right_cheek.x - nose.x,
    ]


def extract_features(hands: list, face_landmarks) -> list[float] | None:
    """
    Fixed-length feature vector for training / ML inference.
    Two hands (sorted left→right by wrist x), each wrist-relative x,y, plus face cues.
    """
    if not hands:
        return None

    ordered = sorted(hands, key=lambda h: h.landmark[0].x)
    rows: list[float] = []
    for i in range(2):
        if i < len(ordered):
            rows.extend(pad_features(landmarks_to_row(ordered[i])))
        else:
            rows.extend([0.0] * FEATURES_PER_HAND)
    rows.extend(face_feature_vector(face_landmarks))
    assert len(rows) == FEATURE_DIM
    return rows


def _rgba_to_bgr_white(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        return img
    bgr = img[:, :, :3]
    alpha = img[:, :, 3:4].astype(np.float32) / 255.0
    white = np.full_like(bgr, 255)
    return (bgr.astype(np.float32) * alpha + white.astype(np.float32) * (1 - alpha)).astype(
        np.uint8
    )


def load_sticker(path: Path) -> list[np.ndarray]:
    path = Path(path)
    if not path.exists():
        return []

    if path.suffix.lower() == ".gif":
        try:
            from PIL import Image

            frames: list[np.ndarray] = []
            gif = Image.open(path)
            idx = 0
            while True:
                frame = gif.convert("RGBA")
                arr = np.array(frame)
                rgba = arr.copy()
                rgba[:, :, [0, 2]] = rgba[:, :, [2, 0]]
                frames.append(_rgba_to_bgr_white(rgba))
                idx += 1
                try:
                    gif.seek(idx)
                except EOFError:
                    break
            if frames:
                return frames
        except Exception:
            pass

        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if frames:
            return frames

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return []
    return [_rgba_to_bgr_white(img)]


def load_all_stickers() -> dict[int, list[np.ndarray]]:
    stickers: dict[int, list[np.ndarray]] = {}
    for cls, name in STICKER_FILES.items():
        frames = load_sticker(ASSETS / name)
        if frames:
            stickers[cls] = frames
    return stickers


def fit_sticker(frames: list[np.ndarray], size: tuple[int, int], frame_idx: int) -> np.ndarray:
    w, h = size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    if not frames:
        cv2.putText(
            canvas,
            "No sticker",
            (40, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (80, 80, 80),
            2,
            cv2.LINE_AA,
        )
        return canvas

    img = frames[frame_idx % len(frames)]
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih) * 0.92
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def finger_extended(hand_landmarks, tip: int, pip: int) -> bool:
    return hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y


def palm_center(hand_landmarks) -> tuple[float, float]:
    # Average wrist + index/middle/pinky MCP
    ids = [0, 5, 9, 13, 17]
    xs = [hand_landmarks.landmark[i].x for i in ids]
    ys = [hand_landmarks.landmark[i].y for i in ids]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def is_open_palm(hand_landmarks) -> bool:
    return all(
        finger_extended(hand_landmarks, tip, pip)
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
    )


def is_ok_sign(hand_landmarks) -> bool:
    """OK sign: thumb tip near index tip; middle+ring+pinky up."""
    lm = hand_landmarks.landmark
    tip_close = _dist(lm[4], lm[8]) < 0.075
    others_up = (
        finger_extended(hand_landmarks, 12, 10)
        and finger_extended(hand_landmarks, 16, 14)
        and finger_extended(hand_landmarks, 20, 18)
    )
    return tip_close and others_up


def is_index_pointing_up(hand_landmarks) -> bool:
    """Only index extended upward; middle/ring/pinky curled."""
    index_up = finger_extended(hand_landmarks, 8, 6)
    others_down = (
        not finger_extended(hand_landmarks, 12, 10)
        and not finger_extended(hand_landmarks, 16, 14)
        and not finger_extended(hand_landmarks, 20, 18)
    )
    return index_up and others_down


def is_crossed_arms(hands: list) -> bool:
    """
    Arms crossed on chest: two hands, wrists crossed / on opposite sides,
    similar height, not raised as open palms.
    """
    if len(hands) < 2:
        return False
    h0, h1 = hands[0], hands[1]
    w0, w1 = h0.landmark[0], h1.landmark[0]
    c0 = palm_center(h0)
    c1 = palm_center(h1)

    # Similar height (folded across chest)
    if abs(c0[1] - c1[1]) > 0.18:
        return False
    # Not too high (near face) — that's pointers / cheek
    if min(c0[1], c1[1]) < 0.28:
        return False
    # Hands somewhat apart horizontally but not wide salute
    dx = abs(c0[0] - c1[0])
    if dx < 0.06 or dx > 0.45:
        return False

    # Crossed: each wrist is on the "wrong" side relative to palm centers,
    # OR wrists are closer than palms (forearms overlapping)
    wrist_dx = abs(w0.x - w1.x)
    palm_dx = abs(c0[0] - c1[0])
    crossed_overlap = wrist_dx < palm_dx * 0.85
    # Mirror-friendly: left-hand palm appears on right side of frame vs other
    side_swap = (c0[0] - c1[0]) * (w0.x - w1.x) < 0

    if not (crossed_overlap or side_swap or wrist_dx < 0.14):
        return False

    # Prefer curled / not both fully open
    if is_open_palm(h0) and is_open_palm(h1):
        return False
    return True


def is_double_pointers_near_face(hands: list, face_landmarks) -> bool:
    """Two index fingers pointing up beside the face (Cartmanland pose)."""
    if face_landmarks is None or len(hands) < 2:
        return False
    if not all(is_index_pointing_up(h) for h in hands[:2]):
        return False

    nose = face_landmarks.landmark[NOSE]
    cheeks = [face_landmarks.landmark[LEFT_CHEEK], face_landmarks.landmark[RIGHT_CHEEK]]
    near_count = 0
    for h in hands[:2]:
        tip = h.landmark[8]
        cx, cy = palm_center(h)
        # Tip or palm near cheek / beside nose
        dists = [
            ((tip.x - a.x) ** 2 + (tip.y - a.y) ** 2) ** 0.5
            for a in [nose, *cheeks]
        ]
        palm_d = min(
            ((cx - a.x) ** 2 + (cy - a.y) ** 2) ** 0.5 for a in [nose, *cheeks]
        )
        if min(dists) < 0.22 or palm_d < 0.24:
            near_count += 1
        # Tips should be roughly at face height
        if abs(tip.y - nose.y) > 0.25:
            return False
    return near_count >= 2


def is_prayer_or_clasp_one_hand(hand_landmarks, face_landmarks=None) -> bool:
    """
    Fallback when MediaPipe merges clasped hands into ONE landmark set.
    Looks like fingers bundled / pointing up near center — NOT beside the face.
    """
    lm = hand_landmarks.landmark
    tips = [lm[i] for i in (8, 12, 16, 20)]
    xs = [t.x for t in tips]
    spread = max(xs) - min(xs)
    if spread > 0.12:
        return False
    if sum(t.y < lm[0].y for t in tips) < 3:
        return False
    if is_ok_sign(hand_landmarks):
        return False
    if is_open_palm(hand_landmarks):
        return False

    # Don't treat cheek-cup as prayer: hand beside face → not begging
    if face_landmarks is not None:
        cx, cy = palm_center(hand_landmarks)
        nose = face_landmarks.landmark[NOSE]
        left_cheek = face_landmarks.landmark[LEFT_CHEEK]
        right_cheek = face_landmarks.landmark[RIGHT_CHEEK]
        near_cheek = min(
            ((cx - left_cheek.x) ** 2 + (cy - left_cheek.y) ** 2) ** 0.5,
            ((cx - right_cheek.x) ** 2 + (cy - right_cheek.y) ** 2) ** 0.5,
        )
        if near_cheek < 0.18:
            return False
        # Prayer is usually centered under the face / chest, not off to one side
        if abs(cx - nose.x) > 0.12 and abs(cy - nose.y) < 0.20:
            return False
    return True


def is_hand_cupping_cheek(hand_landmarks, face_landmarks) -> bool:
    """Hand near cheek — cupping face (anime sticker)."""
    if face_landmarks is None:
        return False
    if is_ok_sign(hand_landmarks):
        return False

    cx, cy = palm_center(hand_landmarks)
    nose = face_landmarks.landmark[NOSE]
    cheeks = [
        face_landmarks.landmark[LEFT_CHEEK],
        face_landmarks.landmark[RIGHT_CHEEK],
    ]
    eye_sides = [
        face_landmarks.landmark[LEFT_EYE_OUTER],
        face_landmarks.landmark[RIGHT_EYE_OUTER],
    ]
    chin = face_landmarks.landmark[CHIN]

    near_cheek = any(((cx - c.x) ** 2 + (cy - c.y) ** 2) ** 0.5 < 0.20 for c in cheeks)
    near_side = any(((cx - p.x) ** 2 + (cy - p.y) ** 2) ** 0.5 < 0.16 for p in eye_sides)
    near_chin_side = ((cx - chin.x) ** 2 + (cy - chin.y) ** 2) ** 0.5 < 0.18
    if not (near_cheek or near_side or near_chin_side):
        return False

    # Beside face (not dead-center prayer under chin)
    beside_face = abs(cx - nose.x) > 0.035
    mid_face_y = abs(cy - nose.y) < 0.28 or abs(cy - chin.y) < 0.22
    if not (beside_face and mid_face_y):
        return False

    # Cupping usually has some curl; allow open-ish palm on cheek too
    return True


def is_mouth_open(face_landmarks) -> bool:
    mouth = _dist(face_landmarks.landmark[LIP_UPPER], face_landmarks.landmark[LIP_LOWER])
    # Normalize by face size (eye outer distance)
    face_w = _dist(
        face_landmarks.landmark[LEFT_EYE_OUTER], face_landmarks.landmark[RIGHT_EYE_OUTER]
    )
    if face_w < 1e-6:
        return False
    return (mouth / face_w) > 0.28


def is_eyes_closed(face_landmarks) -> bool:
    def ear(up, low, outer, inner) -> float:
        vert = _dist(face_landmarks.landmark[up], face_landmarks.landmark[low])
        horiz = _dist(face_landmarks.landmark[outer], face_landmarks.landmark[inner])
        if horiz < 1e-6:
            return 1.0
        return vert / horiz

    left = ear(LEFT_EYE_UP, LEFT_EYE_LOW, LEFT_EYE_OUTER, LEFT_EYE_INNER)
    right = ear(RIGHT_EYE_UP, RIGHT_EYE_LOW, RIGHT_EYE_OUTER, RIGHT_EYE_INNER)
    return left < 0.18 and right < 0.18


def is_begging_hands(hands: list, face_landmarks=None) -> bool:
    """
    Two hands criss-cross / clasped like begging.
    Also accepts one-hand 'merged clasp' that MediaPipe often returns.
    """
    if len(hands) >= 2:
        h0, h1 = hands[0], hands[1]
        w0, w1 = h0.landmark[0], h1.landmark[0]
        wrist_d = _dist(w0, w1)
        c0 = palm_center(h0)
        c1 = palm_center(h1)
        palm_d = ((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2) ** 0.5
        height_diff = abs(c0[1] - c1[1])
        close = wrist_d < 0.22 and palm_d < 0.20 and height_diff < 0.15
        if close:
            both_wide_open = is_open_palm(h0) and is_open_palm(h1)
            if not (both_wide_open and palm_d > 0.12):
                return True
        return False

    if len(hands) == 1:
        # Disabled: one-hand "prayer" was stealing cheek-cup.
        # Begging needs two hands (or ML) now.
        return False
    return False


def is_two_palms_up(hands: list) -> bool:
    if len(hands) < 2:
        return False
    if not (is_open_palm(hands[0]) and is_open_palm(hands[1])):
        return False
    c0 = palm_center(hands[0])
    c1 = palm_center(hands[1])
    avg_y = (c0[1] + c1[1]) / 2
    palm_d = ((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2) ** 0.5
    return avg_y < 0.72 and palm_d > 0.12


def classify_pose(
    hands: list,
    handedness_labels: list[str],
    face_landmarks,
) -> int | None:
    """
    Priority:
      1 crying  — palms up + mouth open + eyes closed
      2 buddha  — OK sign
      3 anime   — hand cupping cheek  (before begging so face poses win)
      0 begging — clasped / prayer
    """
    n = len(hands)
    if n == 0:
        return None

    if (
        face_landmarks is not None
        and is_two_palms_up(hands)
        and is_mouth_open(face_landmarks)
        and is_eyes_closed(face_landmarks)
    ):
        return 1

    if any(is_ok_sign(h) for h in hands):
        return 2

    # Cheek-cup BEFORE begging — cupped hand looks like "prayer" otherwise
    if face_landmarks is not None:
        for h in hands:
            if is_hand_cupping_cheek(h, face_landmarks):
                return 3

    if is_begging_hands(hands, face_landmarks):
        return 0

    return None


# Back-compat name used by older app snippets
def classify_hand_rules(hand_landmarks, handedness_label: str = "Right") -> int | None:
    return classify_pose([hand_landmarks], [handedness_label], None)
