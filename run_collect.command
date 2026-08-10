#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
echo "Opening Collect window — click buttons 4 and 5 to record new poses"
echo "Press q in the camera window when done"
python collect.py
echo
echo "If you recorded classes 4 and 5, run:  python train.py && python app.py"
read -r -p "Press Enter to close..."
