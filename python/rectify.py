"""Board rectification and cell extraction. Mirrors src/vision.cpp."""
import cv2
import numpy as np

from board import AREA_MM, DICT, MARKER_MM, WARP

CELL_PX = WARP // 3
INPUT = 32
INNER_MARGIN = 0.12

MIN_MARKERS = 2


def _params():
    p = cv2.aruco.DetectorParameters()
    p.minMarkerPerimeterRate = 0.01
    p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 3, 53, 4
    p.perspectiveRemovePixelPerCell = 8
    p.polygonalApproxAccuracyRate = 0.05
    return p


_detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), _params())


def _marker_model() -> dict[int, np.ndarray]:
    """Corners of each marker in rectified-board pixels (playing area = [0, WARP]^2)."""
    s, m, a = WARP / AREA_MM, MARKER_MM, AREA_MM
    origins = {0: (-m, -m), 1: (a, -m), 2: (a, a), 3: (-m, a)}
    return {i: np.float32([[x, y], [x + m, y], [x + m, y + m], [x, y + m]]) * s for i, (x, y) in origins.items()}


MARKER_MODEL = _marker_model()


def find_homography(gray: np.ndarray):
    """Image -> rectified board. Every detected marker contributes its 4 corners, so the board
    is still read when up to two markers are hidden by a hand or cut by the frame."""
    corners, ids, _ = _detector.detectMarkers(gray)
    if ids is None:
        return None
    found = {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners) if int(i) in MARKER_MODEL}
    if len(found) < MIN_MARKERS:
        return None
    src = [found[i] for i in sorted(found)]
    dst = [MARKER_MODEL[i] for i in sorted(found)]
    H, _ = cv2.findHomography(np.concatenate(src), np.concatenate(dst), 0)
    return H


def rectify(gray: np.ndarray, H: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(gray, H, (WARP, WARP), flags=cv2.INTER_AREA)


def normalize(cell: np.ndarray) -> np.ndarray:
    """Ink map in [0, 1]: darkness relative to the local paper colour, independent of lighting."""
    cell = cell.astype(np.float32)
    flat = cell.ravel()
    k = int(0.9 * (flat.size - 1))
    paper = float(np.partition(flat, k)[k])
    return np.clip((paper - cell) / max(paper, 1.0), 0, 1) * 2.0


def cells(warped: np.ndarray) -> np.ndarray:
    m = int(INNER_MARGIN * CELL_PX)
    out = []
    for i in range(9):
        y, x = (i // 3) * CELL_PX, (i % 3) * CELL_PX
        crop = warped[y + m:y + CELL_PX - m, x + m:x + CELL_PX - m]
        out.append(normalize(cv2.resize(crop, (INPUT, INPUT), interpolation=cv2.INTER_AREA)))
    return np.stack(out)


def read(image: np.ndarray):
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H = find_homography(gray)
    if H is None:
        return None
    return cells(rectify(gray, H)), H
