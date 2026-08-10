"""
train.py — train a RandomForest on collected landmark CSVs.

Reads data/0.csv … data/3.csv → models/gesture_model.pkl
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from utils import CLASS_NAMES, DATA, FEATURE_DIM, MODELS, NUM_CLASSES, ensure_dirs


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    frames = []
    for label in range(NUM_CLASSES):
        path = DATA / f"{label}.csv"
        if not path.exists():
            print(f"  missing {path.name}")
            continue
        df = pd.read_csv(path)
        if "label" not in df.columns:
            raise SystemExit(f"{path} missing 'label' column — re-run collect.py")
        feat_cols = [c for c in df.columns if c != "label"]
        if len(feat_cols) != FEATURE_DIM:
            raise SystemExit(
                f"{path} has {len(feat_cols)} features, expected {FEATURE_DIM}. "
                "Delete data/*.csv and re-run collect.py"
            )
        frames.append(df)

    if not frames:
        raise SystemExit("No data found. Run: python collect.py")

    data = pd.concat(frames, ignore_index=True)
    y = data["label"].to_numpy()
    X = data.drop(columns=["label"]).to_numpy(dtype=np.float32)
    return X, y


def main() -> None:
    ensure_dirs()
    X, y = load_dataset()
    counts = {i: int((y == i).sum()) for i in range(NUM_CLASSES)}
    print("Samples per class:")
    for i in range(NUM_CLASSES):
        print(f"  {i} ({CLASS_NAMES[i]}): {counts[i]}")

    present = [i for i, n in counts.items() if n > 0]
    if len(present) < 2:
        raise SystemExit("Need samples from at least 2 classes. Run collect.py longer.")

    low = [CLASS_NAMES[i] for i in present if counts[i] < 40]
    if low:
        print(f"\nWarning: low samples for {low} — accuracy may be weak. Aim for ~200 each.")

    min_count = min(counts[i] for i in present)
    strat = y if min_count >= 2 and len(y) >= 10 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    if len(X_test):
        pred = clf.predict(X_test)
        labels = sorted(np.unique(y))
        names = [CLASS_NAMES.get(i, str(i)) for i in labels]
        print("\nClassification report:")
        print(classification_report(y_test, pred, labels=labels, target_names=names))

    out = MODELS / "gesture_model.pkl"
    with out.open("wb") as f:
        pickle.dump({"model": clf, "feature_dim": FEATURE_DIM, "classes": CLASS_NAMES}, f)
    print(f"\nSaved model → {out}")
    print("Run:  python app.py")


if __name__ == "__main__":
    main()
