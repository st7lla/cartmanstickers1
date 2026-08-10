"""
collect.py — record landmark samples for each gesture class.

Controls (click the OpenCV window first!)
-----------------------------------------
  Click buttons on screen  OR  press keys:

  0 / a   begging
  1 / s   crying
  2 / d   buddha
  3 / f   anime
  4 / g   crossed arms
  5 / h   pointer fingers near face
  space   stop recording
  c       clear current class
  q       quit

Aim for ~200 samples per class, then: python train.py
"""

from __future__ import annotations

import csv
import time

import cv2
import mediapipe as mp
import numpy as np

from utils import (
    CLASS_NAMES,
    DATA,
    FEATURE_DIM,
    NUM_CLASSES,
    ensure_dirs,
    extract_features,
)

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

MIN_INTERVAL_SEC = 0.04
TARGET_PER_CLASS = 200
WINDOW = "Collect — Cartman Poses"

# On-screen buttons: (label, x0, y0, x1, y1)
BUTTONS: list[tuple[int, int, int, int, int]] = []


def append_row(label: int, row: list[float]) -> None:
    path = DATA / f"{label}.csv"
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([f"f{i}" for i in range(FEATURE_DIM)] + ["label"])
        writer.writerow(row + [label])


def count_samples(label: int) -> int:
    path = DATA / f"{label}.csv"
    if not path.exists():
        return 0
    with path.open() as f:
        return max(0, sum(1 for _ in f) - 1)


def clear_class(label: int) -> None:
    path = DATA / f"{label}.csv"
    if path.exists():
        path.unlink()


def build_buttons(frame_w: int) -> None:
    global BUTTONS
    BUTTONS = []
    bw, bh = 88, 42
    gap = 6
    total = NUM_CLASSES * bw + (NUM_CLASSES - 1) * gap
    x0 = max(4, (frame_w - total) // 2)
    y0 = 8
    for i in range(NUM_CLASSES):
        x = x0 + i * (bw + gap)
        BUTTONS.append((i, x, y0, x + bw, y0 + bh))


def draw_buttons(frame: np.ndarray, active: int | None) -> None:
    short = {0: "0 beg", 1: "1 cry", 2: "2 OK", 3: "3 cup", 4: "4 Xarm", 5: "5 pts"}
    for label, x0, y0, x1, y1 in BUTTONS:
        on = active == label
        # Highlight new classes
        color = (0, 200, 0) if on else ((0, 120, 200) if label >= 4 else (60, 60, 60))
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 2)
        cv2.putText(
            frame,
            short.get(label, str(label)),
            (x0 + 6, y0 + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def hit_button(x: int, y: int) -> int | None:
    for label, x0, y0, x1, y1 in BUTTONS:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return label
    return None


class MouseState:
    clicked: int | None = None


def on_mouse(event, x, y, flags, param) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        hit = hit_button(x, y)
        if hit is not None:
            MouseState.clicked = hit


def parse_key(key: int) -> str | None:
    """Map waitKey / waitKeyEx codes to actions."""
    if key < 0:
        return None
    key = key & 0xFF
    # Number row
    if key in (ord("0"), ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
        return chr(key)
    # Letter aliases (more reliable on some Mac keyboards)
    aliases = {
        ord("a"): "0",
        ord("s"): "1",
        ord("d"): "2",
        ord("f"): "3",
        ord("g"): "4",
        ord("h"): "5",
    }
    if key in aliases:
        return aliases[key]
    if key in (ord("q"), ord("Q")):
        return "q"
    if key in (ord("c"), ord("C")):
        return "c"
    if key == ord(" "):
        return " "
    # Numpad 0-3 often come through as same as number row after & 0xFF
    return None


def main() -> None:
    ensure_dirs()
    for label in range(NUM_CLASSES):
        path = DATA / f"{label}.csv"
        if not path.exists():
            continue
        with path.open() as f:
            header = f.readline().strip().split(",")
        if len(header) != FEATURE_DIM + 1:
            print(f"Removing outdated {path.name}")
            path.unlink()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check camera permissions.")

    print("=" * 50)
    print("CLICK the camera window, then:")
    print("  Click 0-5 buttons  OR  press 0-5 / a s d f g h")
    print("  4 = crossed arms | 5 = two pointer fingers by face")
    print("  space = stop | c = clear | q = quit")
    print("=" * 50)

    active_label: int | None = None
    last_save = 0.0
    samples_cache = {i: count_samples(i) for i in range(NUM_CLASSES)}
    saving_flash = 0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1000, 700)
    try:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    cv2.setMouseCallback(WINDOW, on_mouse)

    # Bring window to front on macOS (agent-launched windows often hide behind Cursor)
    try:
        import subprocess

        subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to set frontmost of '
                'first process whose name contains "Python" to true',
            ],
            check=False,
            capture_output=True,
        )
    except Exception:
        pass

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.5,
    ) as hands, mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if not BUTTONS:
                build_buttons(w)

            # Mouse button click → select / toggle class
            if MouseState.clicked is not None:
                clicked = MouseState.clicked
                MouseState.clicked = None
                if active_label == clicked:
                    active_label = None
                    print("Stopped recording")
                else:
                    active_label = clicked
                    print(f"Recording class {active_label} ({CLASS_NAMES[active_label]})", flush=True)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hand_result = hands.process(rgb)
            face_result = face_mesh.process(rgb)
            rgb.flags.writeable = True

            hand_list = list(hand_result.multi_hand_landmarks or [])
            face_lms = (
                face_result.multi_face_landmarks[0]
                if face_result.multi_face_landmarks
                else None
            )

            for hand_lms in hand_list:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            status = ""
            now = time.time()
            if active_label is not None:
                if not hand_list:
                    status = "Show your hand(s) to the camera"
                elif (now - last_save) >= MIN_INTERVAL_SEC:
                    # Soft hints only — still record so training isn't blocked
                    if active_label in (0, 1) and len(hand_list) < 2:
                        status = "Tip: use BOTH hands for this pose"
                    elif active_label in (4, 5) and len(hand_list) < 2:
                        status = "Tip: BOTH hands for crossed / pointers"
                    elif active_label in (5, 3) and face_lms is None:
                        status = "Tip: keep your FACE in frame"
                    feats = extract_features(hand_list, face_lms)
                    if feats is not None:
                        append_row(active_label, feats)
                        samples_cache[active_label] = samples_cache.get(active_label, 0) + 1
                        last_save = now
                        saving_flash = 6

            # Green border while actively saving
            if active_label is not None:
                border = (0, 255, 0) if saving_flash > 0 else (0, 180, 255)
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border, 6)
                if saving_flash > 0:
                    saving_flash -= 1

            draw_buttons(frame, active_label)

            counts = "  ".join(f"{i}:{samples_cache[i]}" for i in range(NUM_CLASSES))
            cv2.putText(
                frame,
                f"REC: {active_label if active_label is not None else 'off'}  "
                f"{CLASS_NAMES.get(active_label, '')}",
                (12, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                counts,
                (12, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "CLICK 0-5 / a s d f g h | space=stop | q=quit",
                (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            if status:
                cv2.putText(
                    frame,
                    status,
                    (12, 144),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(WINDOW, frame)
            key = parse_key(cv2.waitKeyEx(1))

            if key == "q":
                break
            if key in ("0", "1", "2", "3", "4", "5"):
                active_label = int(key)
                print(f"Recording class {active_label} ({CLASS_NAMES[active_label]})")
            elif key == " ":
                active_label = None
                print("Stopped recording")
            elif key == "c" and active_label is not None:
                clear_class(active_label)
                samples_cache[active_label] = 0
                print(f"Cleared class {active_label}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nSamples collected:")
    for i in range(NUM_CLASSES):
        n = count_samples(i)
        mark = "OK" if n >= TARGET_PER_CLASS else "-"
        print(f"  [{mark}] {i} ({CLASS_NAMES[i]}): {n}")
    print("\nNext:  python train.py")


if __name__ == "__main__":
    main()
