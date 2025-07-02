from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import easyocr
import re
import numpy as np
import streamlit as st
from shapely.geometry import Polygon, box
from shapely.prepared import PreparedGeometry, prep
from ultralytics import YOLO

# ---------- CONFIG ---------------------------------------------------------

CAR_MODEL_PATH = Path("models/yolo11n.pt")        # детектор + трекер машин
PLATE_MODEL_PATH = Path("models/plate_detector.pt")
POLYGONS_CFG: Dict[str, Dict[str, Any]] = {
    "barrier 1": {
        "points": [(1595, 740), (907, 652), (7, 1255), (1503, 1502)],
        "color": (0, 0, 255),
    },
    "barrier 2": {
        "points": [(460, 823), (161, 704), (2, 783), (7, 1117)],
        "color": (255, 0, 0),
    },
}
TARGET_FPS = 10.0
CLASSES_CAR = [2, 3, 5, 7]                       # COCO: car, motorbike, bus, truck
CONF_CAR = 0.6
CONF_PLATE = 0.6
FRAMES_INSIDE_THRESHOLD = 5                      # сколько кадров подряд в ROI

# ---------- ROI-утилиты ----------------------------------------------------


@dataclass
class ROIState:
    """Состояние одной зоны контроля."""
    name: str
    poly: Polygon
    prep: PreparedGeometry
    bbox: Tuple[float, float, float, float]
    color: Tuple[int, int, int]
    car_id: Optional[int] = None
    frames_inside: int = 0
    last_box: Optional[Tuple[int, int, int, int]] = None


def build_rois(raw: Dict[str, Dict[str, Any]]) -> Dict[str, ROIState]:
    """Преобразует конфиг в готовые к работе ROI."""
    rois: Dict[str, ROIState] = {}
    for name, info in raw.items():
        poly = Polygon(info["points"])
        rois[name] = ROIState(
            name=name,
            poly=poly,
            prep=prep(poly),
            bbox=poly.bounds,
            color=tuple(info["color"]),
        )
    return rois


def _bbox_outside_roi(
    x1: float, y1: float, x2: float, y2: float,
    roi_bbox: Tuple[float, float, float, float],
) -> bool:
    """True → bbox не может пересекать ROI (быстрая фильтрация)."""
    minx, miny, maxx, maxy = roi_bbox
    return x2 < minx or x1 > maxx or y2 < miny or y1 > maxy


def update_rois(
    boxes: np.ndarray,
    track_ids: List[Optional[int]],
    rois: Dict[str, ROIState],
    overlap: float = 0.2,
) -> None:
    """Обновляет car_id / frames_inside / last_box у каждой ROI."""
    for roi in rois.values():
        roi_found = False
        for (x1, y1, x2, y2), tid in zip(boxes, track_ids):
            if tid is None or _bbox_outside_roi(x1, y1, x2, y2, roi.bbox):
                continue
            det_poly = box(float(x1), float(y1), float(x2), float(y2))
            if not roi.prep.intersects(det_poly):
                continue
            inter_area = det_poly.intersection(roi.poly).area
            if inter_area / det_poly.area >= overlap:
                if tid == roi.car_id:
                    roi.frames_inside += 1
                else:
                    roi.car_id = tid
                    roi.frames_inside = 1
                roi.last_box = (int(x1), int(y1), int(x2), int(y2))
                roi_found = True
                break
        if not roi_found:
            roi.car_id = None
            roi.frames_inside = 0
            roi.last_box = None


# ---------- OCR-утилиты ----------------------------------------------------

PLATE_REGEX = re.compile(r'^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$')
LETTER_FIX = {'0': 'О', '1': 'Т', '8': 'В', '4': 'А'}


def fix_plate_text(text: str) -> str:
    """0→О, 1→Т, 8→В, 4→А в буквен­ных позициях."""
    chars = list(text.upper())
    if chars and chars[0] in LETTER_FIX:
        chars[0] = LETTER_FIX[chars[0]]
    if len(chars) >= 6:
        for pos in (4, 5):
            if chars[pos] in LETTER_FIX:
                chars[pos] = LETTER_FIX[chars[pos]]
    return "".join(chars)


def ocr_plate_number(crop_plate: np.ndarray, reader: easyocr.Reader) -> Optional[str]:
    """Возвращает строку гос-номера или None."""
    gray = cv2.cvtColor(crop_plate, cv2.COLOR_BGR2GRAY)
    res = reader.readtext(
        gray, detail=1, allowlist='АВЕКМНОРСТУХ0123456789',
        text_threshold=0.6, low_text=0.4, blocklist=''
    )
    if not res:
        return None
    text = "".join(r[1] for r in reversed(res)).strip()
    text = fix_plate_text(text)
    return text if PLATE_REGEX.match(text) else None


def detect_plate(
    crop_car: np.ndarray,
    plate_detector: YOLO,
    reader: easyocr.Reader,
) -> Optional[str]:
    """Детектирует номер на вырезке машины и делает OCR."""
    result = plate_detector.predict(crop_car, conf=CONF_PLATE,
                                    verbose=False)[0]
    if not result.boxes:
        return None
    for bx in result.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, bx)
        crop_plate = crop_car[y1:y2, x1:x2].copy()
        plate = ocr_plate_number(crop_plate, reader)
        if plate:
            return plate
    return None


# ---------- Основной конвейер (с колбэком) ---------------------------------


def run_video_with_roi_and_plate_stream(
    source: Path | str | int,
    polygons_cfg: Dict[str, Dict[str, Any]],
    callback: callable[[np.ndarray, str], None],
) -> None:
    """
    Обрабатывает видео и каждый кадр вызывает `callback(annotated_frame, status)`.
    Status — общий текст (например, 'barrier 1: id=5, plate=А123ВС45').
    """
    # 1) подготовка
    rois = build_rois(polygons_cfg)
    car_det = YOLO(str(CAR_MODEL_PATH))#.to('cuda:0').half()
    plate_det = YOLO(str(PLATE_MODEL_PATH))#.to('cuda:0').half()
    ocr = easyocr.Reader(["ru"], gpu=True)

    # stride под целевой fps
    cap = cv2.VideoCapture(str(source))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    stride = max(1, round(fps_in / TARGET_FPS))

    plate_registry: Dict[int, str] = {}
    stream = car_det.track(
        source=str(source), stream=True, conf=CONF_CAR,
        classes=CLASSES_CAR, vid_stride=stride, verbose=False,
    )

    for result in stream:
        frame = result.orig_img
        annotated = result.plot(line_width=2)

        if result.boxes and result.boxes.is_track:
            boxes = result.boxes.xyxy.cpu().numpy()
            tids = result.boxes.id.int().cpu().tolist()
            update_rois(boxes, tids, rois)
        else:
            update_rois(np.empty((0, 4), np.float32), [], rois)

        status_lines: List[str] = []
        for roi in rois.values():
            pts = np.array(list(roi.poly.exterior.coords)[:-1], np.int32)
            cv2.polylines(annotated, [pts], True, roi.color, 2)
            if roi.car_id is not None:
                tid = roi.car_id
                if (roi.frames_inside > FRAMES_INSIDE_THRESHOLD
                        and tid not in plate_registry
                        and roi.last_box):
                    x1, y1, x2, y2 = roi.last_box
                    crop_car = frame[y1:y2, x1:x2].copy()
                    plate = detect_plate(crop_car, plate_det, ocr)
                    if plate:
                        plate_registry[tid] = plate
                plate_txt = plate_registry.get(tid, "")
                status = f"{roi.name}: id={tid}, frames={roi.frames_inside}"
                if plate_txt:
                    status += f", plate={plate_txt}"
            else:
                status = f"{roi.name}: No cars"
            status_lines.append(status)
            cv2.putText(
                annotated, status,
                (30, 30 + 40 * list(rois).index(roi.name)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, roi.color, 2,
            )

        callback(annotated, " | ".join(status_lines))


# ---------- Streamlit интерфейс -------------------------------------------


def main() -> None:
    st.set_page_config("ANPR-демо", layout="wide")
    st.title("💡 ANPR-детекция у шлагбаума")

    # блок ввода
    uploaded_file = st.file_uploader("Загрузите видео (AVI/MP4)", type=["avi", "mp4"])
    local_path = st.text_input("…или укажите путь на сервере", "video/cvtest.avi")
    start_btn = st.button("▶️ Запустить")

    placeholder_img = st.empty()
    placeholder_txt = st.empty()

    if start_btn:
        # сохраняем файл, если загрузили
        if uploaded_file is not None:
            tmp_path = Path("tmp") / uploaded_file.name
            tmp_path.parent.mkdir(exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())
            src = tmp_path
        else:
            src = Path(local_path)

        # очередь кадров из бэкенда → UI
        q_frame: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        q_text: queue.Queue[str] = queue.Queue(maxsize=2)

        def _cb(frame: np.ndarray, text: str) -> None:
            try:
                q_frame.put_nowait(frame)
                q_text.put_nowait(text)
            except queue.Full:
                pass

        # фоновая нить обработки
        threading.Thread(
            target=run_video_with_roi_and_plate_stream,
            args=(src, POLYGONS_CFG, _cb),
            daemon=True,
        ).start()

        # основной UI-цикл
        while True:
            try:
                frame = q_frame.get(timeout=0.1)
                text = q_text.get_nowait()
            except queue.Empty:
                continue
            placeholder_txt.markdown(f"**Статус:** {text}")
            placeholder_img.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )
            # небольшой sleep, чтобы UI не «захлёбывался»
            time.sleep(0.03)


if __name__ == "__main__":
    main()