"""Synthetic game video for the README: a slowly moving camera, a hand that covers the board
while a mark is drawn, and O moves taken from the C++ engine itself.

Usage: python python/demo_video.py  ->  data/demo_game.mp4  (then run ttt_vision --video on it)
"""
import math
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

from board import render
from synth import CELL, PX, area_corners, draw_mark

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 960, 720, 20
HUMAN_PREFS = [4, 8, 1, 3, 2, 5, 6, 7, 0]


def camera(t: float):
    tilt = math.radians(28 + 10 * math.sin(0.35 * t))
    rot = math.radians(12 * math.sin(0.22 * t))
    cx, cy = W / 2 + 40 * math.sin(0.3 * t), H / 2 + 25 * math.cos(0.25 * t)
    side = 0.62 * min(W, H)
    sq, far = math.cos(tilt), 1 - 0.35 * math.sin(tilt)
    base = np.float32([[-far, -sq], [far, -sq], [1, sq], [-1, sq]]) * side / 2
    R = np.float32([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
    return (base @ R.T + np.float32([cx, cy])).astype(np.float32)


def frame(board_img, t, hand=None, rng=None):
    h, w = board_img.shape
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), camera(t))
    img = np.full((H, W, 3), (92, 110, 128), np.uint8)
    paper = cv2.warpPerspective(cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR), M, (W, H), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    img = np.where(mask[..., None] > 0, paper, img)
    if hand is not None:
        cell, phase = hand
        o = area_corners(PX)[0]
        p = np.float32([[o[0] + (cell % 3 + 0.5) * CELL, o[1] + (cell // 3 + 0.9) * CELL, 1]])
        x, y, z = (M @ p.T).ravel()
        x, y = x / z, y / z + 60 * (1 - math.sin(math.pi * phase))
        cv2.ellipse(img, (int(x), int(y + 120)), (85, 150), 15, 0, 360, (120, 150, 200), -1)
        cv2.circle(img, (int(x), int(y)), 14, (60, 60, 60), -1)
    gx = np.linspace(0.9, 1.08, W)[None, :, None]
    img = (img.astype(np.float32) * gx + rng.normal(0, 3, img.shape)).clip(0, 255).astype(np.uint8)
    return img


def ai_move(board_img, t, tmp):
    cv2.imwrite(str(tmp), frame(board_img, t, rng=np.random.default_rng(0)))
    out = subprocess.run([str(ROOT / "build" / "ttt_vision"), "--image", str(tmp), "--no-window",
                          "--model", str(ROOT / "models" / "cellnet.onnx")], capture_output=True, text=True).stdout
    m = re.search(r"cell (\d)", out)
    return int(m.group(1)) if m else None


def winner(cells):
    lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    return any(cells[a] and cells[a] == cells[b] == cells[c] for a, b, c in lines)


def main():
    rng = np.random.default_rng(7)
    board_img = render(PX)
    o = area_corners(PX)[0]
    cells = [0] * 9
    out = ROOT / "data" / "demo_game.mp4"
    out.parent.mkdir(exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    t = 0.0

    def emit(n, hand_cell=None):
        nonlocal t
        for k in range(n):
            hand = (hand_cell, k / max(n - 1, 1)) if hand_cell is not None else None
            writer.write(frame(board_img, t, hand, rng))
            t += 1 / FPS

    emit(30)
    turn = 1
    while not winner(cells) and 0 in cells:
        if turn == 1:
            cell = next(c for c in HUMAN_PREFS if not cells[c])
        else:
            cell = ai_move(board_img, t, ROOT / "data" / "_frame.jpg")
        emit(24, cell)
        draw_mark(board_img, (o[0] + (cell % 3) * CELL, o[1] + (cell // 3) * CELL), turn, rng)
        cells[cell] = turn
        emit(40)
        turn = 3 - turn
    emit(40)
    writer.release()
    (ROOT / "data" / "_frame.jpg").unlink(missing_ok=True)
    print("wrote", out, "".join(".XO"[c] for c in cells))


if __name__ == "__main__":
    main()
