from __future__ import annotations
import cv2
import numpy as np
from typing import Callable, Dict

from ultralytics import YOLO
import easyocr
from threading import Event

from src.config import (CAR_MODEL_PATH, PLATE_MODEL_PATH, POLYGONS_CFG,
                     TARGET_FPS, CONF_CAR, CLASSES_CAR,
                     FRAMES_INSIDE_THRESHOLD)
from src.roi import build_rois, update_rois
from src.ocr import detect_plate


def run_video_stream(
    source: str,
    on_frame: Callable[[np.ndarray, str], None],
    stop_event: Event,
) -> None:
    """Основной цикл: детект-трек → ROI → OCR → callback."""
    # модели
    car_det   = YOLO(str(CAR_MODEL_PATH))
    plate_det = YOLO(str(PLATE_MODEL_PATH))
    reader    = easyocr.Reader(["ru"], gpu=True)

    rois = build_rois(POLYGONS_CFG)
    plate_registry: Dict[int, str] = {}

    # stride
    cap = cv2.VideoCapture(str(source))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    stride = max(1, round(fps_in / TARGET_FPS))

    stream = car_det.track(
        source=str(source), stream=True,
        conf=CONF_CAR, classes=CLASSES_CAR, vid_stride=stride,
        verbose=False
    )
    
    try:
        for res in stream:
            if stop_event.is_set():
                break
            frame = res.orig_img
            annotated = res.plot(line_width=2)

            if res.boxes and res.boxes.is_track:
                boxes = res.boxes.xyxy.cpu().numpy()
                tids  = res.boxes.id.int().cpu().tolist()
                update_rois(boxes, tids, rois)
            else:
                update_rois(np.empty((0, 4), np.float32), [], rois)

            status_parts: list[str] = []
            for roi in rois.values():
                pts = np.array(list(roi.poly.exterior.coords)[:-1], np.int32)
                cv2.polylines(annotated, [pts], True, roi.color, 2)
                if roi.car_id is not None:
                    tid = roi.car_id
                    # if roi.frames_inside > FRAMES_INSIDE_THRESHOLD \
                    #         and tid not in plate_registry and roi.last_box:
                    #     x1, y1, x2, y2 = roi.last_box
                    #     car_crop = frame[y1:y2, x1:x2].copy()
                    #     plate = detect_plate(car_crop, plate_det, reader)
                    #     if plate:
                    #         plate_registry[tid] = plate
                    plate_txt = plate_registry.get(tid, "")
                    status = f"{roi.name}: id={tid}, f={roi.frames_inside}"
                    if plate_txt:
                        status += f", plate={plate_txt}"
                else:
                    status = f"{roi.name}: No cars"
                status_parts.append(status)
                cv2.putText(
                    annotated, status,
                    (30, 30 + 40 * list(rois).index(roi.name)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, roi.color, 1,
                )

            on_frame(annotated, " | ".join(status_parts))

    finally:
        try:                       
            stream.close()      # закрываем генератор Ultralytics
        except AttributeError:
            pass