"""Evaluation on unseen synthetic photos: cell accuracy, full-board accuracy, detection rate,
and robustness to the camera tilt. Inference goes through OpenCV DNN with the exported ONNX
model, exactly like the C++ application.

Usage: python python/evaluate.py   (after train.py)
"""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rectify import read  # noqa: E402
from synth import sample  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TEST_DIR = ROOT / "data" / "test"
NET = cv2.dnn.readNetFromONNX(str(ROOT / "models" / "cellnet.onnx"))
CLASSES = ["empty", "X", "O"]


def predict(cells: np.ndarray) -> np.ndarray:
    NET.setInput(cells[:, None].astype(np.float32))
    return NET.forward().argmax(1)


def run(seeds, tilt=None, save=0, hard=False):
    found = cells_ok = boards_ok = 0
    confusion = np.zeros((3, 3), int)
    for k, s in enumerate(seeds):
        img, labels, _ = sample(int(s), tilt, hard)
        if k < save:
            cv2.imwrite(str(TEST_DIR / f"photo_{s}.jpg"), img)
            (TEST_DIR / f"photo_{s}.txt").write_text("".join(".XO"[v] for v in labels))
        r = read(img)
        if r is None:
            continue
        found += 1
        pred = predict(r[0])
        cells_ok += int((pred == labels).sum())
        boards_ok += int((pred == labels).all())
        np.add.at(confusion, (labels, pred), 1)
    n = len(seeds)
    return {"photos": n, "detection_rate": found / n, "cell_accuracy": cells_ok / max(9 * found, 1),
            "board_accuracy_detected": boards_ok / max(found, 1), "board_accuracy_all": boards_ok / n,
            "confusion": confusion.tolist()}


def main():
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    overall = run(range(200_000, 201_000), save=60)
    print("overall", {k: v for k, v in overall.items() if k != "confusion"})
    hard = run(range(400_000, 401_000), hard=True)
    print("hard", {k: v for k, v in hard.items() if k != "confusion"})
    sweep = {}
    for tilt in (0, 15, 30, 45, 55, 65):
        sweep[tilt] = run(range(300_000 + tilt * 1000, 300_000 + tilt * 1000 + 200), tilt=tilt)
        print("tilt", tilt, {k: round(v, 3) for k, v in sweep[tilt].items() if k != "confusion"})
    (RESULTS / "evaluation.json").write_text(json.dumps({"overall": overall, "hard": hard, "tilt_sweep": sweep}, indent=2))

    t = list(sweep)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(t, [sweep[x]["detection_rate"] for x in t], "o-", color="#7a8a99", label="board detected")
    ax.plot(t, [sweep[x]["board_accuracy_detected"] for x in t], "s-", color="#2b6cb0",
            label="all 9 cells correct (when detected)")
    ax.set_xlabel("camera tilt (degrees)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Robustness to the viewing angle (200 photos per angle)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "tilt_robustness.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
