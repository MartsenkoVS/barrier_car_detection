import queue
import threading
from threading import Event
import time
from pathlib import Path
import cv2

import streamlit as st

import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.pipeline import run_video_stream
from src.downloader import download_yadisk


st.set_page_config("Barrier car detection", layout="wide")
st.title("Детекция машин в зоне шлагбаума")
st.markdown("Тестовое видео уже загружено на сервер, можно сразу нажать 'Старт',\
            или для загрузки видео ввести ссылку на Яндекс диск (загрузка ~5 минут), и нажать 'Старт':")

link = st.text_input("Ссылка на видео на Яндекс диске:")
local  = st.text_input("Запуск видео с сервера:", "video/cvtest.avi")
start  = st.button("Старт")

img_pl = st.empty()
txt_pl = st.empty()

if start:
    if link.strip():
        src = download_yadisk(link.strip(), Path("video/cvtest.avi"))
    elif local:
        src = Path(local)
    else:
        st.error("Нужно выбрать файл или указать ссылку!")
        st.stop()

    q_img = queue.Queue(maxsize=2)
    q_txt = queue.Queue(maxsize=2)

    stop_event = Event()

    def _cb(frame, text) -> None:           # callback из конвейера
        try:
            q_img.put_nowait(frame)
            q_txt.put_nowait(text)
        except queue.Full:
            pass

    # запускаем поток и «привязываем» его к сессии Streamlit
    t = threading.Thread(
        target=run_video_stream,
        args=(src, _cb, stop_event),
        daemon=True)
    t.start()

    try:
        # отображаем кадры, пока поток жив
        while t.is_alive():
            try:
                frame = q_img.get(timeout=0.1)
                text  = q_txt.get_nowait()
            except queue.Empty:
                continue
            txt_pl.markdown(f"**Статус:** {text}")
            img_pl.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            time.sleep(0.03)
    finally:
        stop_event.set()
        t.join(timeout=2)