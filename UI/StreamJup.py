import os
#? Note that these env variables are for linux. The current terminal outputs OpenH264 library error but can be ignored for now
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # AV_LOG_QUIET

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from ultralytics import YOLO

st.set_page_config(page_title="Smart Vertical Basketball Streaming", layout="wide")
st.title("Smart Vertical Basketball Streaming")

@st.cache_resource
def load_model():
    return YOLO("../models/yolov8m.pt")

def draw_boxes(frame, boxes, confidences):
    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{conf:.0%}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return frame

def process_video(video_path, model):
    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    annotated_frames = []
    frame_counts     = []
    progress         = st.progress(0, text="Running detection...")
    start            = time.time()

    for i, result in enumerate(model.predict(source=video_path, classes=[0], stream=True, save=False, conf=0.50, iou=0.95)):
        frame_rgb   = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
        boxes       = result.boxes.xyxy.cpu().numpy() if result.boxes else []
        confidences = result.boxes.conf.cpu().numpy() if result.boxes else []

        annotated_frames.append(draw_boxes(frame_rgb.copy(), boxes, confidences))
        frame_counts.append(len(boxes))

        if total_frames > 0:
            progress.progress(min((i + 1) / total_frames, 1.0),
                              text=f"Frame {i+1}/{total_frames} - {len(boxes)} people detected")

    progress.empty()
    return annotated_frames, frame_counts, fps, time.time() - start

def render_video(frames, fps):
    """Write annotated RGB frames to a temp mp4 using H.264 (avc1) for browser playback."""
    h, w = frames[0].shape[:2]

    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = out_tmp.name
    out_tmp.close()

    fourcc = cv2.VideoWriter.fourcc(*"avc1")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        st.error("H.264 (avc1) codec not available. Install ffmpeg and add it to PATH.")
        return None

    progress = st.progress(0, text="Rendering video...")
    for i, frame_rgb in enumerate(frames):
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        progress.progress((i + 1) / len(frames), text=f"Rendering frame {i+1}/{len(frames)}")
    writer.release()
    progress.empty()
    return out_path


uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if st.button("Run Detection", type="primary"):
        frames, counts, fps, elapsed = process_video(tmp_path, load_model())
        os.unlink(tmp_path)

        st.session_state.frames  = frames
        st.session_state.counts  = counts
        st.session_state.elapsed = elapsed
        st.session_state.fps     = fps

if "frames" in st.session_state:
    frames  = st.session_state.frames
    counts  = st.session_state.counts
    elapsed = st.session_state.elapsed
    fps     = st.session_state.fps

    st.success(f"Processed {len(frames)} frames in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total frames",     len(frames))
    col2.metric("Avg people/frame", f"{np.mean(counts):.1f}")
    col3.metric("Peak people",      max(counts) if counts else 0)

    st.divider()

    tab_scrubber, tab_play = st.tabs(["Frame Scrubber", "Play Video"])

    with tab_scrubber:
        frame_idx = st.slider("Scrub through frames", 0, len(frames) - 1, 0)
        st.image(frames[frame_idx],
                 caption=f"Frame {frame_idx} - {counts[frame_idx]} people",
                 width='stretch')

    with tab_play:
        if st.button("Render & Play Video"):
            out_path = render_video(frames, fps)
            if out_path is not None:
                with open(out_path, "rb") as f:
                    st.session_state.video_bytes = f.read()
                os.unlink(out_path)

        if "video_bytes" in st.session_state:
            st.video(st.session_state.video_bytes)

else:
    st.info("Upload a video to get started.")