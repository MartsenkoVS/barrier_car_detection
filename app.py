from pathlib import Path
from typing import Generator

import cv2
import time
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO

# Конфигурация
VIDEO_PATH = Path("video/cvtest.avi")  # Путь к видео
TARGET_FPS = 30  # Целевой FPS стрима
BOUNDARY = "--frame"  # Разделитель в MJPEG


app = FastAPI()
model = YOLO("models/yolo11n.pt")


HTML_PAGE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Детекция машин — MJPEG</title>
</head>
<body>
    <h2>Стрим с детекцией — 30 fps (MJPEG)</h2>
    <!-- просто вставляем src, браузер сам рендерит JPEG-поток -->
    <img src="/video_feed" width="640" alt="Stream">
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """
    Возвращает простую HTML-страницу с <img src="/video_feed">.
    """
    return HTMLResponse(HTML_PAGE)


def mjpeg_generator(src: Path) -> Generator[bytes, None, None]:
    """
    Синхронный генератор, читающий кадры из видео, детектирующий,
    кодирующий в JPEG и отдающий в формате multipart/x-mixed-replace.
    """
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {src}")

    # исходный FPS и шаг пропуска кадров
    in_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    skip = max(1, round(in_fps / TARGET_FPS))
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % skip != 0:
                continue

            # трекинг/детекция
            results = model.track(
                frame, conf=0.5, classes=[2, 3, 5, 7], persist=True
            )
            annotated = results[0].plot()

            # JPEG-кодирование
            success, buffer = cv2.imencode('.jpg', annotated)
            if not success:
                continue

            jpg_bytes = buffer.tobytes()

            # Собираем chunk для MJPEG
            yield (
                BOUNDARY.encode() + b"\r\n"
                + b"Content-Type: image/jpeg\r\n\r\n"
                + jpg_bytes
                + b"\r\n"
            )

            # Контролируем fps
            time.sleep(1.0 / TARGET_FPS)
    finally:
        cap.release()


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    """
    Эндпоинт для MJPEG-стрима.
    """
    return StreamingResponse(
        mjpeg_generator(VIDEO_PATH),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY.lstrip('--')}"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1,
    )