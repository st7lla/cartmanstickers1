"""
app.py — Pose App (ML + gesture overrides).

  0 begging   — clasped hands
  1 crying    — palms + open mouth + closed eyes
  2 buddha    — OK sign
  3 anime     — hand cupping cheek
  4 crossed   — arms crossed on chest
  5 pointers  — two index fingers up by face

Keys: q quit
"""

from __future__ import annotations

import pickle
import time

import cv2
import mediapipe as mp
import numpy as np

from utils import (
    CLASS_NAMES,
    FEATURE_DIM,
    LEFT_CHEEK,
    MODELS,
    NOSE,
    RIGHT_CHEEK,
    ensure_dirs,
    extract_features,
    fit_sticker,
    is_crossed_arms,
    is_double_pointers_near_face,
    is_ok_sign,
    load_all_stickers,
    palm_center,
)

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

FACE_DOT_IDX = [33, 133, 362, 263, 1, 4, 61, 291, 13, 14, 234, 454]

CONFIRM_FRAMES = 3
HOLD_SEC = 0.45
ML_MIN_PROB = 0.30


def load_model():
    path = MODELS / "gesture_model.pkl"
    if not path.exists():
        return None
    with path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        if obj.get("feature_dim") != FEATURE_DIM:
            print("Bad model feature size — retrain.")
            return None
        model = obj["model"]
        # Old 4-class model can't predict 4/5
        classes = set(int(c) for c in model.classes_)
        if not {4, 5}.issubset(classes):
            print("Model missing new classes 4/5 — using rules until you retrain.")
            return None
        return model
    return obj


def near_face(hand, face, thresh: float = 0.20) -> bool:
    if face is None:
        return False
    anchors = [face.landmark[NOSE], face.landmark[LEFT_CHEEK], face.landmark[RIGHT_CHEEK]]
    cx, cy = palm_center(hand)
    checks = [
        (cx, cy),
        (hand.landmark[0].x, hand.landmark[0].y),
        (hand.landmark[9].x, hand.landmark[9].y),
    ]
    for px, py in checks:
        for a in anchors:
            if (px - a.x) ** 2 + (py - a.y) ** 2 < thresh * thresh:
                return True
    return False


def ml_predict(model, hands, face) -> tuple[int | None, float]:
    if model is None or not hands:
        return None, 0.0

    views = [hands]
    if len(hands) > 1:
        for h in hands:
            views.append([h])

    best_label, best_p = None, 0.0
    for view in views:
        feats = extract_features(view, face)
        if feats is None:
            continue
        probs = model.predict_proba([feats])[0]
        idx = int(np.argmax(probs))
        p = float(probs[idx])
        if p > best_p:
            best_p = p
            best_label = int(model.classes_[idx])
    if best_label is None or best_p < ML_MIN_PROB:
        return None, best_p
    return best_label, best_p


def predict(model, hands, face) -> int | None:
    if not hands:
        return None

    # Geometric overrides (order matters)
    if any(is_ok_sign(h) for h in hands):
        return 2
    if is_double_pointers_near_face(hands, face):
        return 5
    if is_crossed_arms(hands):
        return 4

    label, conf = ml_predict(model, hands, face)

    # Single hand on cheek → anime (not when two pointer fingers)
    if face is not None and len(hands) == 1 and near_face(hands[0], face):
        if label == 1 and conf > 0.75:
            return 1
        return 3

    return label


def main() -> None:
    ensure_dirs()
    stickers = load_all_stickers()
    if not stickers:
        raise SystemExit("Missing assets/ stickers")

    model = load_model()
    mode = "trained ML" if model is not None else "rules only (collect 4+5 then train)"
    print(f"Pose App — {mode}")
    print("  0 clasp | 1 crying | 2 OK | 3 cup cheek | 4 crossed arms | 5 pointer fingers")
    print("Press q to quit")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam")

    cam_w, cam_h = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    prediction: int | None = None
    candidate: int | None = None
    candidate_count = 0
    hold_until = 0.0
    gif_tick = 0

    idle = np.full((cam_h, cam_w, 3), 255, dtype=np.uint8)
    tips = [
        "0 begging",
        "1 crying",
        "2 buddha OK",
        "3 cup cheek",
        "4 crossed arms",
        "5 pointers by face",
    ]
    for i, tip in enumerate(tips):
        cv2.putText(
            idle,
            tip,
            (160, 160 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (120, 120, 120),
            2,
        )

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands_det, mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (cam_w, cam_h))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hand_res = hands_det.process(rgb)
            face_res = face_det.process(rgb)
            rgb.flags.writeable = True

            face = face_res.multi_face_landmarks[0] if face_res.multi_face_landmarks else None
            hand_list = list(hand_res.multi_hand_landmarks or [])

            if face is not None:
                for idx in FACE_DOT_IDX:
                    lm = face.landmark[idx]
                    cv2.circle(frame, (int(lm.x * cam_w), int(lm.y * cam_h)), 2, (255, 255, 255), -1)

            for h in hand_list:
                mp_drawing.draw_landmarks(
                    frame,
                    h,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            live = predict(model, hand_list, face)

            now = time.time()
            if live is not None:
                if live == candidate:
                    candidate_count += 1
                else:
                    candidate, candidate_count = live, 1
                if candidate_count >= CONFIRM_FRAMES:
                    prediction = int(candidate)
                    hold_until = now + HOLD_SEC
            else:
                candidate, candidate_count = None, 0
                if now > hold_until:
                    prediction = None

            cv2.putText(
                frame,
                f"Prediction: {prediction if prediction is not None else '-'}",
                (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if prediction is not None:
                cv2.putText(
                    frame,
                    CLASS_NAMES.get(prediction, ""),
                    (12, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            gif_tick += 1
            if prediction is not None and int(prediction) in stickers:
                right = fit_sticker(stickers[int(prediction)], (cam_w, cam_h), gif_tick // 2)
            else:
                right = idle

            cv2.imshow("Pose App", np.hstack([frame, right]))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
