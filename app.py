"""
app.py
------
Streamlit app: upload foto atau video coffeeshop, lalu deteksi
'person', 'chair', 'dining table' menggunakan YOLOv8 (pretrained COCO).

Cara jalankan:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
import tempfile
import time
from PIL import Image

from utils.detector import SeatDetector
from utils.occupancy import get_chair_status, summarize_occupancy

st.set_page_config(page_title="Coffeeshop Seat Detector", layout="wide")

# ------------------------------------------------------------------
# Load model sekali saja (cache), biar tidak reload tiap kali interaksi
# ------------------------------------------------------------------
@st.cache_resource
def load_detector(model_name: str, conf: float):
    return SeatDetector(model_path=model_name, conf=conf)


# ------------------------------------------------------------------
# Sidebar - pengaturan
# ------------------------------------------------------------------
st.sidebar.title("⚙️ Pengaturan")

model_choice = st.sidebar.selectbox(
    "Model YOLOv8",
    options=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
    index=0,
    help="n = nano (paling cepat), s = small, m = medium (lebih akurat, lebih berat).",
)

conf_thresh = st.sidebar.slider(
    "Confidence threshold", min_value=0.1, max_value=0.9, value=0.35, step=0.05
)

overlap_thresh = st.sidebar.slider(
    "Overlap threshold untuk status kursi (IoU)",
    min_value=0.05, max_value=0.5, value=0.1, step=0.05,
    help="Semakin kecil, semakin sensitif menganggap kursi 'occupied'.",
)

show_boxes = st.sidebar.multiselect(
    "Tampilkan kelas",
    options=["person", "chair", "dining table"],
    default=["person", "chair", "dining table"],
)

detector = load_detector(model_choice, conf_thresh)

st.title("☕ Coffeeshop Seat & Table Detector")
st.caption("Upload foto atau video coffeeshop untuk mendeteksi orang, kursi, dan meja secara otomatis.")

tab_photo, tab_video = st.tabs(["📷 Foto", "🎥 Video"])

# ------------------------------------------------------------------
# TAB FOTO
# ------------------------------------------------------------------
with tab_photo:
    uploaded_image = st.file_uploader(
        "Upload foto coffeeshop", type=["jpg", "jpeg", "png"], key="image_uploader"
    )

    if uploaded_image is not None:
        pil_img = Image.open(uploaded_image).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        with st.spinner("Mendeteksi objek..."):
            detections = detector.predict(img_bgr)
            detections_filtered = [d for d in detections if d["cls"] in show_boxes]
            result_img = detector.draw(img_bgr, detections_filtered)
            chair_status = get_chair_status(detections, overlap_thresh)
            occ_summary = summarize_occupancy(chair_status)
            cls_summary = detector.summarize(detections)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Gambar Asli")
            st.image(img_rgb)
        with col2:
            st.subheader("Hasil Deteksi")
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))

        st.subheader("📊 Ringkasan")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Orang terdeteksi", cls_summary["person"])
        m2.metric("Kursi terdeteksi", cls_summary["chair"])
        m3.metric("Kursi terisi", occ_summary["occupied"])
        m4.metric("Kursi kosong", occ_summary["empty"])

# ------------------------------------------------------------------
# TAB VIDEO
# ------------------------------------------------------------------
with tab_video:
    uploaded_video = st.file_uploader(
        "Upload video coffeeshop (mp4/mov/avi)", type=["mp4", "mov", "avi"], key="video_uploader"
    )

    frame_skip = st.slider(
        "Proses tiap berapa frame (biar lebih cepat)",
        min_value=1, max_value=30, value=5,
        help="Nilai lebih besar = lebih cepat diproses, tapi update lebih jarang.",
    )

    run_video = st.button("▶️ Mulai Proses Video")

    if uploaded_video is not None and run_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        stframe = st.empty()
        stmetrics = st.empty()
        progress_bar = st.progress(0)

        frame_idx = 0
        last_detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Supaya lebih cepat, kita cuma jalankan model tiap N frame,
            # dan pakai hasil terakhir untuk frame di antaranya.
            if frame_idx % frame_skip == 0:
                last_detections = detector.predict(frame)

            filtered = [d for d in last_detections if d["cls"] in show_boxes]
            drawn = detector.draw(frame, filtered)

            chair_status = get_chair_status(last_detections, overlap_thresh)
            occ_summary = summarize_occupancy(chair_status)
            cls_summary = detector.summarize(last_detections)

            stframe.image(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))

            with stmetrics.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Orang", cls_summary["person"])
                m2.metric("Kursi total", cls_summary["chair"])
                m3.metric("Kursi terisi", occ_summary["occupied"])
                m4.metric("Kursi kosong", occ_summary["empty"])

            frame_idx += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        st.success("Selesai memproses video.")

st.divider()
st.caption(
    "Model: YOLOv8 pretrained COCO (kelas person, chair, dining table). "
    "Status kursi (occupied/empty) dihitung dari overlap bounding box orang & kursi — "
    "logic ini bisa disempurnakan lagi dengan data coffeeshop asli."
)
