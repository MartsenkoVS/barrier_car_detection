from __future__ import annotations

from pathlib import Path
from typing import Iterator
import threading
from queue import Queue, Full, Empty

import cv2
import numpy as np
from collections import OrderedDict
import time

from src.pipeline import run_video_stream
from src.config import MJPEG_BOUNDARY

# Глобальное хранилище для распознанных номеров
plates_live = OrderedDict()

def mjpeg_generator(
    source: Path,
    stop_event: threading.Event,
) -> Iterator[bytes]:
    """
    Запускает run_video_stream в отдельном потоке и собирает
    последние аннотированные кадры в очередь. Каждый кадр
    кодирует в JPEG и отдает как часть MJPEG-потока.
    """
    # Очередь для хранения только самого свежего кадра
    frame_queue: Queue[np.ndarray] = Queue(maxsize=1)

    def on_frame(frame: np.ndarray, status: str) -> None:
        """
        Callback из pipeline: помещает новый кадр в очередь.
        Старые кадры отбрасываются.
        """
        try:
            frame_queue.put_nowait(frame)
        except Full:
            pass  # если очередь полна, пропустить

        # Обновляем список номеров
        # status — строка вида "barrier 1: id=3, f=32, plate=А123ВС77 | barrier 2: No cars"
        parts = status.split(" | ")
        for part in parts:
            if "plate=" in part:
                plate = part.split("plate=")[-1]
                if plate not in plates_live:
                    plates_live[plate] = time.time()

    # Запускаем pipeline в фоновом потоке
    worker = threading.Thread(
        target=run_video_stream,
        args=(str(source), on_frame, stop_event),
        daemon=True,
    )
    worker.start()

    boundary = MJPEG_BOUNDARY.encode()
    
    try:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.1)
            except Empty:
                continue  # ждем следующий кадр

            # Кодируем JPEG
            success, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not success:
                continue
            jpg_bytes = buf.tobytes()

            # Формируем chunk для MJPEG
            yield (
                boundary + b"\r\n"
                + b"Content-Type: image/jpeg\r\n\r\n"
                + jpg_bytes
                + b"\r\n"
            )
    except GeneratorExit:
        raise
    finally:
        # Завершаем работу фонового потока
        plates_live.clear()
        stop_event.set()
        worker.join(timeout=1)