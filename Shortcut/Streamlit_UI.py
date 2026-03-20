import streamlit as st
import cv2
import numpy as np
import tempfile
import os

st.write("Edge detection on video using OpenCV. 🎥")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    # Save uploaded file to a temp file (OpenCV needs a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    st.write(f"**FPS:** {fps:.1f} | **Frames:** {total_frames} | **Duration:** {duration:.1f}s")

    # Scrubber to pick a frame
    frame_idx = st.slider("Select frame", 0, max(total_frames - 1, 0), 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    os.unlink(tmp_path)

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        edges = cv2.Canny(frame_rgb, 100, 200)

        tab1, tab2 = st.tabs(["Detected edges", "Original"])
        tab1.image(edges, use_column_width=True)
        tab2.image(frame_rgb, use_column_width=True)
    else:
        st.error("Could not read the selected frame.")
else:
    st.info("Upload a video file to get started.")