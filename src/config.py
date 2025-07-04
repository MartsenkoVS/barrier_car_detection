from pathlib import Path
from typing import Dict, Any

CAR_MODEL_PATH = Path("models/yolo11n.pt")
PLATE_MODEL_PATH = Path("models/plate_detector.pt")
VIDEO_PATH = Path("video/cvtest.avi")

TARGET_FPS = 30.0
TARGET_WIDTH = 960
MJPEG_BOUNDARY = "--frame" # Разделитель кадров в MJPEG

CLASSES_CAR = [2, 3, 5, 7]     # car, motorbike, bus, truck
CONF_CAR = 0.5
CONF_PLATE = 0.6

FRAMES_INSIDE_THRESHOLD = 30
MISSING_FRAMES_THRESHOLD = 3
ROI_OVERLAP = 0.2

POLYGONS_CFG: Dict[str, Dict[str, Any]] = {
    "barrier 1": {
        "points": [(1595, 740), (907, 652), (7, 1255), (1503, 1502)],
        "color": (0, 0, 255),
        "plate_detection": True
    },
    "barrier 2": {
        "points": [(460, 823), (161, 704), (2, 783), (7, 1117)],
        "color": (255, 0, 0),
        "plate_detection": False
    },
}