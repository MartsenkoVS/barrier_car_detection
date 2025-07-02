from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import re
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from dataclasses import dataclass
from shapely.geometry import Polygon, box
from shapely.prepared import prep, PreparedGeometry
from concurrent.futures import ThreadPoolExecutor


@dataclass
class ROIState:
    name: str
    poly: Polygon
    prep: PreparedGeometry
    bbox: tuple[float, float, float, float]
    color: tuple[int, int, int]

    car_id: int | None = None
    frames_inside: int = 0
    last_box: tuple[int, int, int, int] | None = None


def build_rois(raw: dict[str, dict[str, Any]]) -> dict[str, ROIState]:
    """Создаём готовые к работе зоны из исходных данных."""
    rois: dict[str, ROIState] = {}
    for name, info in raw.items():
        poly = Polygon(info["points"])
        rois[name] = ROIState(
            name=name,
            poly=poly,
            prep=prep(poly),
            bbox=poly.bounds,           # (minx, miny, maxx, maxy)
            color=tuple(info["color"]),
        )
    return rois


def _bbox_outside_roi(x1: float, y1: float, x2: float, y2: float,
                      roi_bbox: tuple[float, float, float, float]) -> bool:
    """True → bbox не может пересекать ROI."""
    minx, miny, maxx, maxy = roi_bbox
    return x2 < minx or x1 > maxx or y2 < miny or y1 > maxy


def update_rois(
    boxes: np.ndarray,
    track_ids: list[int | None],
    rois: dict[str, ROIState],
    overlap: float = 0.2,
) -> None:

    for roi in rois.values():
        roi_found = False
        for (x1, y1, x2, y2), tid in zip(boxes, track_ids):
            if tid is None or _bbox_outside_roi(x1, y1, x2, y2, roi.bbox):
                continue

            det_poly = box(float(x1), float(y1), float(x2), float(y2))

            # ---- булевая проверка на подготовленном полигоне ----
            if not roi.prep.intersects(det_poly):
                continue

            # ---- только если пересечение есть, считаем площадь ----
            inter_area = det_poly.intersection(roi.poly).area
            if inter_area / det_poly.area >= overlap:
                if tid == roi.car_id:
                    roi.frames_inside += 1
                else:
                    roi.car_id = tid
                    roi.frames_inside = 1
                roi.last_box = x1, y1, x2, y2
                roi_found = True
                break

        if not roi_found:
            roi.car_id = None
            roi.frames_inside = 0
            roi.last_box = None


# Регулярка для проверки формата
PLATE_REGEX = re.compile(r'^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$')

# Маппинг для автозамены в буквеных позициях
LETTER_FIX = {
    '0': 'О',
    '1': 'Т',
    '8': 'В',
    '4': 'А',
}


def fix_plate_text(text: str) -> str:
    """
    Автозаменяем цифры на буквы в тех позициях,
    где по формату должны быть буквы: 0→О,1→Т,8→В,4→А.
    """
    chars = list(text)
    # Первая буква:
    if len(chars) >= 1 and chars[0] in LETTER_FIX:
        chars[0] = LETTER_FIX[chars[0]]
    # две буквы после трёх цифр:
    if len(chars) >= 6:
        for pos in (4, 5):
            if chars[pos] in LETTER_FIX:
                chars[pos] = LETTER_FIX[chars[pos]]
    return "".join(chars)


# OCR через EasyOCR
def ocr_plate_number(crop_plate: np.ndarray, ocr_reader: easyocr.Reader):
    gray = cv2.cvtColor(crop_plate, cv2.COLOR_RGB2GRAY)
    ocr_results = ocr_reader.readtext(gray,
                                detail=1,
                                blocklist='',
                                text_threshold=0.6,    # Порог для текста (0.0 - 1.0)
                                low_text=0.4,          # Нижний порог для текста (0.0 - text_threshold))
                                allowlist = 'АВЕКМНОРСТУХ0123456789')
    if ocr_results:
        # объединяем все найденные фрагменты в одну строку
        texts = [res[1] for res in ocr_results]
        texts = list(reversed(texts))
        plate_number = "".join(texts).strip()
        plate_number = fix_plate_text(plate_number)
        if PLATE_REGEX.match(plate_number):
            return plate_number
    return None


def detect_plate(
    crop_car: np.ndarray,
    plate_detector: YOLO,
    ocr_reader: easyocr.Reader,
) -> Optional[str]:
    """
    Детект номера на вырезе и распознаёт текст через EasyOCR.
    Возвращает строку номера или None.
    """
    # 1) Детектируем номер через YOLO (подаём на вход crop)
    results = plate_detector.predict(
        source=crop_car, conf=0.6, verbose=False
    )
    result = results[0]
    if result.boxes:
        boxes = result.boxes.xyxy.cpu().numpy()
        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            crop_plate = crop_car[y1:y2, x1:x2].copy()
            plate_number = ocr_plate_number(crop_plate, ocr_reader)
            if plate_number:
                return plate_number
    else:
        return None


def run_video_with_roi_and_plate(
    source: Union[str, int, Path],
    model_path: Union[str, Path],
    plate_model_path: Union[str, Path],
    polygons_cfg: Dict[str, Dict[str, Any]],
    target_fps: float = 10.0,
    classes_list: Optional[List[int]] = None,
    conf_threshold: float = 0.6,
) -> None:
    """
    Работа с видео/камерой:
    * трекинг машин YOLO
    * проверка пересечения с ROI
    * OCR номеров при «достаточном» пребывании в зоне

    Все изменения против исходной версии касаются только блока ROI.
    """
    classes_list = classes_list or [2, 3, 5, 7]

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть источник: {source!r}")
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    vid_stride: int = max(1, round(input_fps / target_fps))

    car_detector = YOLO(str(model_path))
    plate_detector = YOLO(str(plate_model_path))
    ocr_reader = easyocr.Reader(["ru"], gpu=True)

    plate_registry: Dict[int, str] = {}  # track_id: номер
    rois = build_rois(polygons_cfg)

    executor = ThreadPoolExecutor(max_workers=1)

    def _recognize_async(crop_car: np.ndarray, tid: int):
        plate = detect_plate(crop_car, plate_detector, ocr_reader)
        if plate:
            plate_registry[tid] = plate

    cv2.namedWindow("ANPR with ROI", cv2.WINDOW_NORMAL)

    stream = car_detector.track(
        source=str(source),
        stream=True,
        conf=conf_threshold,
        classes=classes_list,
        vid_stride=vid_stride,
        verbose=False,
    )

    for result in stream:
        frame = result.orig_img
        annotated = result.plot(line_width=2)

        # обновляем зоны по детекциям
        if result.boxes and result.boxes.is_track:
            boxes = result.boxes.xyxy.cpu().numpy()
            tids = result.boxes.id.int().cpu().tolist()
            update_rois(boxes, tids, rois)
        else:
            update_rois(np.empty((0, 4), dtype=np.float32), [], rois)

        # обход по всем ROI
        for idx, roi in enumerate(rois.values()):
            # рисуем контур зоны
            pts = np.array(list(roi.poly.exterior.coords)[:-1], np.int32)
            cv2.polylines(annotated, [pts], isClosed=True,
                          color=roi.color, thickness=2)

            if roi.car_id is not None:
                tid = roi.car_id

                if roi.frames_inside > 5 \
                        and tid not in plate_registry \
                        and roi.last_box is not None:
                    x1, y1, x2, y2 = roi.last_box
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    crop_car = frame[y1:y2, x1:x2].copy()
                    executor.submit(_recognize_async, crop_car, tid)

                plate_text = plate_registry.get(tid, "")
                status = (f"id={tid}, frames={roi.frames_inside}"
                          f"{', plate=' + plate_text if plate_text else ''}")
            else:
                status = "No cars"

            # вывод статуса
            cv2.putText(
                annotated,
                f"{roi.name}: {status}",
                (30, 30 + 50 * idx),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                roi.color,
                2,
            )

        cv2.imshow("ANPR with ROI", annotated)

        # выход по «q»
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    executor.shutdown(wait=True)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    polygons_dict = {
        "barrier 1": {
            "points": [(1595, 740), (907, 652), (7, 1255), (1503, 1502)],
            "color": (0, 0, 255),
        },
        "barrier 2": {
            "points": [(460, 823), (161, 704), (2, 783), (7, 1117)],
            "color": (255, 0, 0),
        },
    }

    run_video_with_roi_and_plate(
        source=Path("video/cvtest.avi"),
        model_path=Path("models/yolo11n.pt"),
        plate_model_path=Path("models/plate_detector.pt"),
        polygons_cfg=polygons_dict,
        target_fps=10.0,
        classes_list=[2, 3, 5, 7],
        conf_threshold=0.5,
    )