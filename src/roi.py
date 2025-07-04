from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
from shapely.geometry import Polygon, box
from shapely.prepared import prep, PreparedGeometry

from src.config import MISSING_FRAMES_THRESHOLD, ROI_OVERLAP


@dataclass
class ROIState:
    name: str
    poly: Polygon
    prep: PreparedGeometry
    bbox: Tuple[float, float, float, float]
    color: Tuple[int, int, int]
    plate_detection: bool
    car_id: Optional[int] = None
    frames_inside: int = 0
    missing_count: int = 0
    last_box: Optional[np.ndarray] = None


def build_rois(
    raw: Dict[str, Dict[str, Any]],
    scale: float = 1.0
) -> Dict[str, ROIState]:
    """
    Масштабирует исходные полигоны в едином масштабе `scale`
    и создаёт ROIState для каждого.
    """
    rois: Dict[str, ROIState] = {}
    for name, info in raw.items():
        pts = [(x * scale, y * scale) for x, y in info["points"]]
        poly = Polygon(pts)
        rois[name] = ROIState(
            name=name,
            poly=poly,
            prep=prep(poly),
            bbox=poly.bounds,
            color=tuple(info["color"]),
            plate_detection=info["plate_detection"],
        )
    return rois


def _bbox_outside_roi(
    x1: float, y1: float, x2: float, y2: float,
    roi_bbox: Tuple[float, float, float, float],
) -> bool:
    minx, miny, maxx, maxy = roi_bbox
    return x2 < minx or x1 > maxx or y2 < miny or y1 > maxy


def update_rois(
    boxes: np.ndarray,
    tids: List[int | None],
    rois: Dict[str, ROIState],
) -> None:
    for roi in rois.values():
        roi_found = False
        for (x1, y1, x2, y2), tid in zip(boxes, tids):
            if tid is None or _bbox_outside_roi(x1, y1, x2, y2, roi.bbox):
                continue
            det_poly = box(float(x1), float(y1), float(x2), float(y2))
            if not roi.prep.intersects(det_poly):
                continue
            if det_poly.intersection(roi.poly).area / roi.poly.area >= ROI_OVERLAP:
                if tid == roi.car_id:
                    roi.frames_inside += 1
                else:
                    roi.car_id = tid
                    roi.frames_inside = 1
                roi.missing_count = 0
                roi.last_box = np.array([x1, y1, x2, y2], dtype=int)
                roi_found = True
                break

        if not roi_found and roi.car_id is not None:
            # машина была в зоне, но на этом кадре пропала
            roi.missing_count += 1
            if roi.missing_count > MISSING_FRAMES_THRESHOLD:
                # превышена толерантность пропуска — сбрасываем состояние
                roi.car_id = None
                roi.frames_inside = 0
                roi.last_box = None
                roi.missing_count = 0