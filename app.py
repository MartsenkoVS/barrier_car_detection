from fastapi import FastAPI
from routers.video_stream import router

app = FastAPI(
    title="Barrier Car Detection API",
    description="Сервис для стриминга видео с детекцией машин в формате MJPEG",
    version="1.0.0",
)

# Подключаем все маршруты из routers/video_stream.py
app.include_router(router)