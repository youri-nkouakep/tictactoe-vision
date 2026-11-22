"""Printable board: a 3x3 grid framed by four ArUco markers whose inner corners
mark the corners of the playing area. Shared geometry for the generator and the pipeline.

Usage: python python/board.py   ->  board/board.pdf and board/board.png
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = (0, 1, 2, 3)          # top-left, top-right, bottom-right, bottom-left
INNER_CORNER = (2, 3, 0, 1)        # corner of each marker touching the playing area
AREA_MM, MARKER_MM, MARGIN_MM = 150.0, 22.0, 4.0
WARP = 300                          # side of the rectified board in pixels (3 cells of 100 px)


def render(px_per_mm: float = 8.0, grid_gray: int = 140) -> np.ndarray:
    """Board as a white image: playing area, grid lines and the four markers around it."""
    m, a = int(MARKER_MM * px_per_mm), int(AREA_MM * px_per_mm)
    pad = int(MARGIN_MM * px_per_mm)
    size = a + 2 * m + 2 * pad
    img = np.full((size, size), 255, np.uint8)
    o = pad + m
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    tops = [(pad, pad), (pad, o + a), (o + a, o + a), (o + a, pad)]
    for mid, (y, x) in zip(MARKER_IDS, tops):
        img[y:y + m, x:x + m] = cv2.aruco.generateImageMarker(dictionary, mid, m)
    t = max(2, int(1.2 * px_per_mm))
    for k in (1, 2):
        c = o + k * a // 3
        cv2.line(img, (c, o), (c, o + a), grid_gray, t)
        cv2.line(img, (o, c), (o + a, c), grid_gray, t)
    cv2.rectangle(img, (o, o), (o + a, o + a), 225, max(1, t // 3))
    return img


def area_corners(px_per_mm: float = 8.0) -> np.ndarray:
    m, a, pad = MARKER_MM * px_per_mm, AREA_MM * px_per_mm, MARGIN_MM * px_per_mm
    o = pad + m
    return np.float32([[o, o], [o + a, o], [o + a, o + a], [o, o + a]])


def write_pdf(path: Path):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    img = render(px_per_mm=12.0)
    png = ROOT / "board" / "board.png"
    cv2.imwrite(str(png), img)
    side = (AREA_MM + 2 * MARKER_MM + 2 * MARGIN_MM) * mm
    W, H = A4
    c = canvas.Canvas(str(path), pagesize=A4)
    c.drawImage(ImageReader(str(png)), (W - side) / 2, (H - side) / 2 + 20 * mm, side, side)
    c.setFont("Helvetica", 10)
    c.drawCentredString(W / 2, (H - side) / 2, "TicTacToe Vision - print at 100 % scale. You play X, the AI plays O.")
    c.save()


if __name__ == "__main__":
    (ROOT / "board").mkdir(exist_ok=True)
    write_pdf(ROOT / "board" / "board.pdf")
    print("board/board.pdf written")
