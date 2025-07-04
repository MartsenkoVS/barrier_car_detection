import re
from typing import Optional

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

from src.config import CONF_PLATE

PLATE_REGEX = re.compile(r'^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$')
LETTER_FIX = {'0': 'O', '1': 'T', '8': 'B', '4': 'A'}
ALLOWLIST = 'ABEKMHOPCTYX0123456789'


def _fix_plate(text: str) -> str:
    chars = list(text.upper())
    if chars and chars[0] in LETTER_FIX:
        chars[0] = LETTER_FIX[chars[0]]
    if len(chars) >= 6:
        for i in (4, 5):
            if chars[i] in LETTER_FIX:
                chars[i] = LETTER_FIX[chars[i]]
    return "".join(chars)


def ocr_plate(crop: np.ndarray, reader: easyocr.Reader) -> Optional[str]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    res = reader.readtext(gray, detail=1, allowlist=ALLOWLIST,
                          text_threshold=0.6, low_text=0.4)
    if not res:
        return None
    text = "".join(r[1] for r in reversed(res)).strip()
    text = _fix_plate(text)
    return text if PLATE_REGEX.match(text) else None


def detect_plate(
    crop_car: np.ndarray,
    plate_det: YOLO,
    reader: easyocr.Reader,
) -> Optional[str]:
    result = plate_det.predict(crop_car, conf=CONF_PLATE, verbose=False)[0]
    if not result.boxes:
        return None
    for bx in result.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, bx)
        crop_plate = crop_car[y1:y2, x1:x2]
        plate = ocr_plate(crop_plate, reader)
        if plate:
            return plate
    return None