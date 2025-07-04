import cv2
import numpy as np
from typing import Tuple


def resize_frame(
    frame: np.ndarray,
    width: int
) -> Tuple[np.ndarray, float]:
    """
    Ресайзит кадр до заданной ширины, сохраняя пропорции.
    Возвращает:
      - resized: кадр с новой шириной,
      - scale: общий коэффициент масштабирования (по X и Y).
    """
    h0, w0 = frame.shape[:2]
    if w0 == width:
        return frame, 1.0

    scale = width / w0
    new_h = int(h0 * scale)
    resized = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale

def scale_boxes(
    boxes: np.ndarray,
    inv_scale: float
) -> np.ndarray:
    """
    Приводит боксы из уменьшенного кадра
    обратно в координаты оригинала.
    """
    arr = np.asarray(boxes, dtype=float)
    scaled = arr * inv_scale
    return scaled.astype(int)