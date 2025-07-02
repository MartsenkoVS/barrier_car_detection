from pathlib import Path
from typing import Generator

import cv2
import time
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO

# Путь к видео и базовые настройки
VIDEO_PATH: Path = Path("video/cvtest.avi")
TARGET_FPS: int = 30
MJPEG_BOUNDARY: str = "--frame"
TARGET_WIDTH: int = 640  # ширина кадра после ресайза

app = FastAPI()

# Загружаем модель и при возможности отправляем её на GPU
model = YOLO("models/yolo11n.pt")
# if torch.cuda.is_available():
#     model.to("cuda")
#     print("Модель переведена на GPU")

# Простая HTML-страница с <img src="/video_feed">
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Детекция машин — MJPEG</title>
</head>
<body>
    <h2>Стрим с детекцией — 30 fps (MJPEG)</h2>
    <img src="/video_feed" width="640" alt="Stream">
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """
    Возвращает HTML-страницу с тегом <img>, который
    сам обновляет кадры через MJPEG-поток.
    """
    return HTMLResponse(HTML_PAGE)


def mjpeg_generator(src: Path) -> Generator[bytes, None, None]:
    """
    Синхронный генератор MJPEG-потока:
    1) читает кадры из src,
    2) ресайзит до TARGET_WIDTH,
    3) детектит машинки,
    4) кодирует в JPEG с качеством 60,
    5) отдаёт в формате multipart/x-mixed-replace.
    """
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {src}")

    # Очищаем буфер, чтобы не тащить застрявшие кадры
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Определяем, сколько кадров пропускать
    in_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    skip = max(1, round(in_fps / TARGET_FPS))
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            # Пропускаем лишние кадры для стабильного TARGET_FPS
            if frame_idx % skip:
                continue

            # Ресайз популярного кадра
            height, width = frame.shape[:2]
            new_height = int(height * TARGET_WIDTH / width)
            frame_resized = cv2.resize(
                frame,
                (TARGET_WIDTH, new_height),
                interpolation=cv2.INTER_LINEAR,
            )

            # Детекция + трекинг
            results = model.track(
                frame_resized,
                conf=0.5,
                classes=[2, 3, 5, 7],
                persist=True,
            )
            annotated = results[0].plot()  # cv2.ndarray

            # Кодируем в JPEG (качество 60)
            success, buffer = cv2.imencode(
                ".jpg",
                annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), 60],
            )
            if not success:
                continue

            jpg_bytes = buffer.tobytes()

            # Формируем chunk для MJPEG
            yield (
                MJPEG_BOUNDARY.encode() + b"\r\n"
                + b"Content-Type: image/jpeg\r\n\r\n"
                + jpg_bytes
                + b"\r\n"
            )

            # Небольшая пауза, чтобы выровнять ~30 fps
            time.sleep(1.0 / TARGET_FPS)
    finally:
        cap.release()


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    """
    Эндпоинт, отдающий MJPEG-поток с заданным boundary.
    """
    return StreamingResponse(
        mjpeg_generator(VIDEO_PATH),
        media_type=(
            f"multipart/x-mixed-replace; boundary="
            f"{MJPEG_BOUNDARY.lstrip('-')}"
        ),
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
