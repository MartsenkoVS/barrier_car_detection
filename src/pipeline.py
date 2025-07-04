from __future__ import annotations
import cv2
import numpy as np
from typing import Callable, Dict

from ultralytics import YOLO
import easyocr
from threading import Event
import time

from src.config import (CAR_MODEL_PATH, PLATE_MODEL_PATH, POLYGONS_CFG,
                     TARGET_FPS, TARGET_WIDTH, CONF_CAR, CLASSES_CAR,
                     FRAMES_INSIDE_THRESHOLD)
from src.roi import build_rois, update_rois
from src.ocr import detect_plate
from src.utils import resize_frame, scale_boxes


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

    plate_registry: Dict[int, str] = {}

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {source}")

    # Очищаем буфер, чтобы не тащить застрявшие кадры
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Определяем, сколько кадров пропускать
    in_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    skip = max(1, round(in_fps / TARGET_FPS))

    # Читаем первый кадр для рассчета параметров
    ret, frame_orig = cap.read()
    if not ret:
        cap.release()
        return
    
    # Ресайз
    frame_resized, scale = resize_frame(frame_orig, TARGET_WIDTH)
    inv_scale = 1 / scale
    rois = build_rois(POLYGONS_CFG, scale)
    frame_idx = 1

    try:
        while not stop_event.is_set() and ret:
            # Пропускаем лишние кадры для стабильного TARGET_FPS
            if frame_idx % skip != 0:
                ret, frame_orig = cap.read()
                frame_idx += 1
                continue
            
            frame_resized, _ = resize_frame(frame_orig, TARGET_WIDTH)
            
            # Детекция + трекинг машин
            results = car_det.track(
                frame_resized,
                conf= CONF_CAR,
                classes=CLASSES_CAR,
                persist=True,
                verbose=False
            )
            res = results[0]
            annotated = res.plot()

            if res.boxes and res.boxes.is_track:
                boxes = res.boxes.xyxy.cpu().numpy()
                tids  = res.boxes.id.int().cpu().tolist()
            else:
                boxes = np.empty((0, 4), dtype=float)
                tids = []
            # Обновляем состояние зон
            update_rois(boxes, tids, rois)

            # OCR и отрисовка
            # # Размеры для отрисовки текста
            # line_h        = 30
            # top_pad       = 10
            # bottom_pad    = 10
            # banner_h      = top_pad + bottom_pad + line_h * len(rois)
            # banner_w      = TARGET_WIDTH

            # # чёрная плашка для текста
            # banner = np.zeros((banner_h, banner_w, 3), dtype=np.uint8)

            status_parts: list[str] = []
            


            for idx, roi in enumerate(rois.values()):
                # рисуем полигоны
                pts = np.array(list(roi.poly.exterior.coords)[:-1], np.int32)
                cv2.polylines(annotated, [pts], True, roi.color, 2)
                if roi.car_id is not None:
                    tid = roi.car_id
                    if (roi.plate_detection
                        and roi.frames_inside > FRAMES_INSIDE_THRESHOLD
                        and tid not in plate_registry
                        and roi.last_box is not None
                    ):
                        # Вырезаем из оригинального кадра
                        x1, y1, x2, y2 = scale_boxes(roi.last_box, inv_scale)
                        car_crop = frame_orig[y1:y2, x1:x2]
                        plate = detect_plate(car_crop, plate_det, reader)
                        if plate:
                            plate_registry[tid] = plate

                    plate_txt = plate_registry.get(tid, "")
                    status = f"{roi.name}: id={tid}, f={roi.frames_inside}, m={roi.missing_count}"
                    if plate_txt:
                        status += f", plate={plate_txt}"
                else:
                    status = f"{roi.name}: No cars"

                status_parts.append(status)

                cv2.putText(
                    annotated, status,
                    (15, 30 + 30 * list(rois).index(roi.name)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, roi.color, 1,
                )
            #     # Рисуем текст
            #     y = top_pad + line_h * (idx + 1) - 8
            #     cv2.putText(
            #         banner,
            #         status,
            #         org=(15, y),
            #         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #         fontScale=1.0,
            #         color=roi.color,
            #         thickness=2,
            #         lineType=cv2.LINE_AA,
            #     )
            
            # annotated = np.vstack((banner, annotated))

            # Передаем в calback
            on_frame(annotated, " | ".join(status_parts))

            # Читаем следующий кадр
            ret, frame_orig = cap.read()
            frame_idx += 1

            # Небольшая пауза, чтобы выровнять ~30 fps
            time.sleep(1.0 / TARGET_FPS)

    finally:
        cap.release()