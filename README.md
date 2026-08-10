# Cartman Stickers — Pose App

HAMPTER-style webcam meme app: **MediaPipe** hand classification + **OpenCV** split view, with Cartman stickers instead of the hamster.

```
┌─────────────────────┬─────────────────────┐
│  Webcam + landmarks │  Cartman sticker    │
│  Prediction: 0      │                     │
└─────────────────────┴─────────────────────┘
```

## Stickers

| Class | Gesture (all conditions required) | Asset | Meme |
|------:|-----------------------------------|-------|------|
| 0 | Two hands clasped / criss-cross (begging) | `assets/0.jpg` | Scheming Cartman |
| 1 | Two palms up **and** mouth open **and** eyes closed | `assets/1.gif` | Crying Cartman |
| 2 | OK hand sign only | `assets/2.png` | Buddha Cartman |
| 3 | Hand cupping cheek | `assets/3.jpg` | Anime meme |
| 4 | Arms crossed on chest | `assets/4.jpg` | Crossed-arms Cartman |
| 5 | Two index fingers up beside face | `assets/5.png` | Pointer-fingers Cartman |

## Setup

```bash
cd cartmanstickers
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

macOS: allow camera access for Terminal/Python in **System Settings → Privacy & Security → Camera**.

## Run (works immediately with rule gestures)

```bash
python app.py
```

Press `q` to quit.

## Optional: train your own poses (HAMPTER pipeline)

1. Collect samples — hold `0`, `1`, or `2` while posing; `space` to stop recording; `q` to quit:

```bash
python collect.py
```

2. Train the classifier:

```bash
python train.py
```

3. Run the app again — it loads `models/gesture_model.pkl` automatically:

```bash
python app.py
```

## Project layout

```
cartmanstickers/
├── app.py              # Pose App (split camera | sticker)
├── collect.py          # Landmark data collection
├── train.py            # RandomForest trainer
├── utils.py            # Shared helpers
├── requirements.txt
├── assets/             # Stickers used by the app (0–5)
├── photos/             # Original source images (not used at runtime)
├── data/               # Collected CSVs (after collect.py)
└── models/             # gesture_model.pkl (after train.py)
```
