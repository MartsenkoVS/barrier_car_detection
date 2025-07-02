import streamlit as st
from src.downloader import download_yadisk
import cv2
from pathlib import Path
import io
from ultralytics import YOLO

model = YOLO("models/yolo11n.pt")


st.set_page_config("Barrier car detection", layout="wide")
st.title("Детекция машин в зоне шлагбаума")
st.markdown("Тестовое видео уже загружено на сервер, можно сразу нажать 'Старт',\
            или для загрузки видео ввести ссылку на Яндекс диск (загрузка ~5 минут), и нажать 'Старт':")

vid_file = st.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
link = st.text_input("Ссылка на видео на Яндекс диске:")
local  = st.text_input("Запуск видео с сервера:", "video/cvtest.avi")
start  = st.button("Старт")

img_pl = st.empty()
txt_pl = st.empty()

if start:
    if vid_file:
        g = io.BytesIO(vid_file.read())  # BytesIO Object
        with open("video/video.mp4", "wb") as out:  # Open temporary file as bytes
            out.write(g.read())  # Read bytes into file
        src = Path("video/video.mp4")
    elif link.strip():
        src = download_yadisk(link.strip(), Path("video/cvtest.avi"))
    elif local:
        src = Path(local)
    else:
        st.error("Нужно выбрать файл или указать ссылку!")
        st.stop()

    if src:
        stop_button = st.button("Stop")  # Button to stop the inference
        cap = cv2.VideoCapture(src)  # Capture the video
        if not cap.isOpened():
            st.error("Could not open webcam or video source.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                st.warning("Failed to read frame from webcam. Please verify the webcam is connected properly.")
                break

            # Process frame with model
            results = model.track(
                frame, conf=0.5, classes=[2, 3, 5, 7], persist=True
            )

            annotated_frame = results[0].plot()  # Add annotations on frame

            if stop_button:
                cap.release()  # Release the capture
                st.stop()  # Stop streamlit app

            img_pl.image(annotated_frame, channels="BGR", width=640)  # Display processed frame

        cap.release()  # Release the capture
    cv2.destroyAllWindows()  # Destroy all OpenCV windows