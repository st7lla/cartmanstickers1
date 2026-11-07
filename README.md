# cartman stickers 

```
┌─────────────────────┬─────────────────────┐
│  Webcam + landmarks │  Cartman sticker    │
│  Prediction: 0      │                     │
└─────────────────────┴─────────────────────┘
```

## stickers

| Class | Gesture (all conditions required) | Asset | Meme |
|------:|-----------------------------------|-------|------|
| 0 | Two hands clasped / criss-cross (begging) | `assets/0.jpg` | Scheming Cartman |
| 1 | Two palms up **and** mouth open **and** eyes closed | `assets/1.gif` | Crying Cartman |
| 2 | OK hand sign only | `assets/2.png` | Buddha Cartman |
| 3 | Hand cupping cheek | `assets/3.jpg` | Anime meme |
| 4 | Arms crossed on chest | `assets/4.jpg` | Crossed-arms Cartman |
| 5 | Two index fingers up beside face | `assets/5.png` | Pointer-fingers Cartman |

## setup

```bash
cd cartmanstickers
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

macOS: allow camera access for Terminal/Python in **System Settings → Privacy & Security → Camera**.

## run (works immediately with rule gestures)

```bash
python app.py
```

Press `q` to quit.

## optional: train your own poses 

1. collect samples — hold `0`, `1`, or `2` while posing; `space` to stop recording; `q` to quit:

```bash
python collect.py
```

2. train the classifier:

```bash
python train.py
```

3. run the app again — it loads `models/gesture_model.pkl` automatically:

```bash
python app.py
```

## project layout

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
