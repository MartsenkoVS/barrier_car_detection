from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
from shapely.geometry import Polygon, box
from shapely.prepared import prep, PreparedGeometry


@dataclass
class ROIState:
    name: str
    poly: Polygon
    prep: PreparedGeometry
    bbox: Tuple[float, float, float, float]
    color: Tuple[int, int, int]
    car_id: Optional[int] = None
    frames_inside: int = 0
    last_box: Optional[Tuple[int, int, int, int]] = None


def build_rois(raw: Dict[str, Dict[str, Any]]) -> Dict[str, ROIState]:
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
    minx, miny, maxx, maxy = roi_bbox
    return x2 < minx or x1 > maxx or y2 < miny or y1 > maxy


def update_rois(
    boxes: np.ndarray,
    tids: List[int | None],
    rois: Dict[str, ROIState],
    overlap: float = 0.2,
) -> None:
    for roi in rois.values():
        roi_found = False
        for (x1, y1, x2, y2), tid in zip(boxes, tids):
            if tid is None or _bbox_outside_roi(x1, y1, x2, y2, roi.bbox):
                continue
            det_poly = box(float(x1), float(y1), float(x2), float(y2))
            if not roi.prep.intersects(det_poly):
                continue
            if det_poly.intersection(roi.poly).area / roi.poly.area >= overlap:
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