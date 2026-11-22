"""Synthetic photos of the board with hand-drawn X and O, and their ground truth.

Each photo varies the drawing (pen, size, wobble, position in the cell), the camera
(perspective tilt, rotation, distance) and the image (lighting gradient, shadow, blur,
noise, JPEG compression). Labels: 0 empty, 1 X, 2 O.
"""
import math
import random

import cv2
import numpy as np

from board import area_corners, render

PX = 8.0
CELL = int(150 * PX / 3)


def _wobbly(points, rng, amount):
    pts = np.asarray(points, np.float32)
    dense = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(2, int(np.linalg.norm(b - a) / 6))
        for t in np.linspace(0, 1, n, endpoint=False):
            dense.append(a + t * (b - a))
    dense.append(pts[-1])
    dense = np.array(dense)
    noise = np.cumsum(rng.normal(0, amount, dense.shape), axis=0)
    noise -= np.linspace(0, 1, len(dense))[:, None] * noise[-1]
    return (dense + noise).astype(np.int32)


def draw_mark(img, cell_xy, mark, rng: np.random.Generator, hard: bool = False):
    x0, y0 = cell_xy
    size = CELL * (rng.uniform(0.3, 0.9) if hard else rng.uniform(0.42, 0.78))
    off = 0.2 if hard else 0.12
    cx = x0 + CELL / 2 + rng.uniform(-off, off) * CELL
    cy = y0 + CELL / 2 + rng.uniform(-off, off) * CELL
    color = int(rng.uniform(90, 150)) if hard else int(rng.uniform(10, 70))
    thick = int(rng.uniform(3, 7)) if hard else int(rng.uniform(5, 16))
    wob = rng.uniform(0.3, 1.6)
    if mark == 1:
        tilt = rng.uniform(-0.25, 0.25)
        for sign in (1, -1):
            a = (cx - size / 2 * math.cos(tilt) * sign, cy - size / 2)
            b = (cx + size / 2 * math.cos(tilt) * sign, cy + size / 2)
            jitter = rng.normal(0, size * 0.05, 4)
            pts = _wobbly([(a[0] + jitter[0], a[1] + jitter[1]), (b[0] + jitter[2], b[1] + jitter[3])], rng, wob)
            cv2.polylines(img, [pts], False, color, thick, cv2.LINE_AA)
    else:
        rx, ry = size / 2 * rng.uniform(0.8, 1.05), size / 2 * rng.uniform(0.8, 1.05)
        start = rng.uniform(0, 2 * math.pi)
        sweep = rng.uniform(1.85, 2.15) * math.pi
        ts = np.linspace(start, start + sweep, 60)
        pts = [(cx + rx * math.cos(t), cy + ry * math.sin(t)) for t in ts]
        cv2.polylines(img, [_wobbly(pts, rng, wob)], False, color, thick, cv2.LINE_AA)


def board_with_marks(labels, rng, hard=False):
    img = render(PX)
    o = area_corners(PX)[0]
    for i, lab in enumerate(labels):
        if lab:
            draw_mark(img, (o[0] + (i % 3) * CELL, o[1] + (i // 3) * CELL), lab, rng, hard)
    return img


def random_state(rng):
    """A position reachable in a real game: X starts, counts differ by at most one."""
    n = int(rng.integers(0, 10))
    order = rng.permutation(9)[:n]
    labels = np.zeros(9, int)
    for k, i in enumerate(order):
        labels[i] = 1 if k % 2 == 0 else 2
    return labels


def photograph(board, rng, tilt=None, out=(960, 720), hard=False):
    """Place the board on a table and view it through a random camera."""
    h, w = board.shape
    W, H = out
    table = np.full((H, W, 3), rng.integers(60, 200, 3), np.uint8)
    table = (table + rng.normal(0, 12, (H, W, 3))).clip(0, 255).astype(np.uint8)
    paper = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    paper = (paper.astype(np.float32) * rng.uniform(0.85, 1.0, 3)).clip(0, 255).astype(np.uint8)

    tilt = rng.uniform(0, 45) if tilt is None else tilt
    scale = (rng.uniform(0.3, 0.45) if hard else rng.uniform(0.45, 0.75)) * min(W, H) / h
    side = h * scale
    cx, cy = W / 2 + rng.uniform(-0.12, 0.12) * W, H / 2 + rng.uniform(-0.1, 0.1) * H
    rot = math.radians(rng.uniform(-30, 30))
    squeeze = math.cos(math.radians(tilt))
    far = 1 - 0.35 * math.sin(math.radians(tilt))
    base = np.float32([[-far, -squeeze], [far, -squeeze], [1, squeeze], [-1, squeeze]]) * side / 2
    R = np.float32([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
    dst = (base @ R.T + np.float32([cx, cy])).astype(np.float32)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(paper, M, (W, H), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    img = np.where(mask[..., None] > 0, warped, table)

    gx, gy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    angle = rng.uniform(0, 2 * math.pi)
    light = 1 + rng.uniform(0.1, 0.45) * (math.cos(angle) * gx + math.sin(angle) * gy) / 2
    if rng.random() < 0.4:
        sx, sy, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.2, 0.5) * W
        light *= 1 - 0.35 * np.exp(-((gx * W / 2 + W / 2 - sx) ** 2 + (gy * H / 2 + H / 2 - sy) ** 2) / (2 * r * r))
    img = (img.astype(np.float32) * light[..., None] * rng.uniform(0.7, 1.15)).clip(0, 255)
    k = int(rng.choice([5, 7])) if hard else int(rng.choice([1, 3, 3, 5]))
    if k > 1:
        img = cv2.GaussianBlur(img, (k, k), 0)
    img = (img + rng.normal(0, rng.uniform(2, 9), img.shape)).clip(0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 90))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), M @ np.vstack([area_corners(PX).T, np.ones(4)]), tilt


def sample(seed: int, tilt=None, hard: bool = False):
    """hard=True: pencil-light thin strokes, off-centre marks, a smaller board and stronger blur."""
    rng = np.random.default_rng(seed)
    random.seed(seed)
    labels = random_state(rng)
    photo, _, tilt = photograph(board_with_marks(labels, rng, hard), rng, tilt, hard=hard)
    return photo, labels, tilt
