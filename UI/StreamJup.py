import os
#? Note that these env variables are for linux. The current terminal outputs OpenH264 library error but can be ignored for now
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # AV_LOG_QUIET

import functools
import json
import socket
import tempfile
import threading
import time
import http.server

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from ultralytics import YOLO

st.set_page_config(page_title="Smart Vertical Basketball Streaming", layout="wide")
st.title("Smart Vertical Basketball Streaming")


# ─── Local video server ───────────────────────────────────────────────────────
# Serves the rendered video file over HTTP so the browser can fetch it directly,
# avoiding the Streamlit websocket size limit.

_active_server: dict = {}   # holds {"server": HTTPServer, "path": str, "url": str}

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def _start_video_server(video_path: str) -> str:
    """Shut down any previous server, then serve video_path on a new port."""
    global _active_server
    if _active_server:
        _active_server["server"].shutdown()
        try:
            os.unlink(_active_server["path"])
        except FileNotFoundError:
            pass
        _active_server = {}

    directory = os.path.dirname(video_path)
    filename  = os.path.basename(video_path)
    port      = _free_port()

    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass  # silence request logs

    handler = functools.partial(_QuietHandler, directory=directory)
    server = http.server.HTTPServer(("localhost", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f"http://localhost:{port}/{filename}"
    _active_server = {"server": server, "path": video_path, "url": url}
    return url


# ─── Model ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return YOLO("../models/yolov8m.pt")


# ─── Helpers ─────────────────────────────────────────────────────────────────

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
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    clean_frames     = []
    annotated_frames = []
    frame_counts     = []
    frame_boxes      = {}     # {frame_index: [[x1,y1,x2,y2,conf], ...]}
    progress         = st.progress(0, text="Running detection...")
    start            = time.time()

    for i, result in enumerate(model.predict(
            source=video_path, classes=[0], stream=True,
            save=False, conf=0.50, iou=0.95)):
        frame_rgb   = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
        boxes       = result.boxes.xyxy.cpu().numpy() if result.boxes else np.empty((0, 4))
        confidences = result.boxes.conf.cpu().numpy() if result.boxes else np.empty(0)

        clean_frames.append(frame_rgb.copy())
        annotated_frames.append(draw_boxes(frame_rgb.copy(), boxes, confidences))
        frame_counts.append(int(len(boxes)))

        if len(boxes):
            frame_boxes[str(i)] = [
                [float(x1), float(y1), float(x2), float(y2), float(c)]
                for (x1, y1, x2, y2), c in zip(boxes, confidences)
            ]

        if total_frames > 0:
            progress.progress(min((i + 1) / total_frames, 1.0),
                              text=f"Frame {i+1}/{total_frames} — {len(boxes)} people detected")

    progress.empty()
    return (clean_frames, annotated_frames, frame_counts,
            frame_boxes, frame_w, frame_h, fps, time.time() - start)


def render_video(frames, fps, label="Rendering video") -> str | None:
    """Write RGB frames to a *persistent* temp file (H.264). Returns the file path."""
    h, w = frames[0].shape[:2]

    # delete=False so the file stays alive for the HTTP server to serve
    out_tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = out_tmp.name
    out_tmp.close()

    fourcc = cv2.VideoWriter.fourcc(*"avc1")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        st.error("H.264 (avc1) codec unavailable. Install ffmpeg and add it to PATH.")
        os.unlink(out_path)
        return None

    progress = st.progress(0, text=label)
    for i, frame_rgb in enumerate(frames):
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        progress.progress((i + 1) / len(frames),
                          text=f"{label} — frame {i+1}/{len(frames)}")
    writer.release()
    progress.empty()
    return out_path


def build_player_html(video_url: str, frame_boxes: dict,
                      frame_w: int, frame_h: int, fps: float) -> str:
    boxes_json = json.dumps(frame_boxes)
    return f"""
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #000; }}

  #player-wrap {{
    position: relative;
    width: 100%;
    background: #000;
  }}
  #vid {{
    width: 100%;
    height: auto;    /* natural height — no letterboxing */
    display: block;
  }}
  #overlay {{
    position: absolute;
    top: 0; left: 0;
    pointer-events: none;
  }}
  #controls {{
    height: 44px;
    display: flex;
    align-items: center;
    padding: 4px 8px;
    background: #111;
  }}
  #toggleBtn {{
    padding: 5px 18px;
    font-size: 13px;
    font-weight: bold;
    border: 2px solid #00cc44;
    border-radius: 6px;
    background: #00cc44;
    color: #fff;
    cursor: pointer;
  }}
  #toggleBtn.off {{
    background: #333;
    border-color: #555;
    color: #aaa;
  }}
</style>

<div id="controls">
  <button id="toggleBtn" onclick="toggleBoxes()">Boxes: ON</button>
</div>
<div id="player-wrap">
  <video id="vid" controls>
    <source src="{video_url}" type="video/mp4">
  </video>
  <canvas id="overlay"></canvas>
</div>

<script>
  const BOXES  = {boxes_json};
  const FPS    = {fps};
  const VID_W  = {frame_w};
  const VID_H  = {frame_h};
  let showBoxes = true;

  const vid    = document.getElementById('vid');
  const canvas = document.getElementById('overlay');
  const ctx    = canvas.getContext('2d');

  function resizeCanvas() {{
    // Match canvas exactly to the video's rendered pixel area (no letterbox offset)
    canvas.width        = vid.clientWidth;
    canvas.height       = vid.clientHeight;
    canvas.style.width  = vid.clientWidth  + 'px';
    canvas.style.height = vid.clientHeight + 'px';
  }}

  function drawFrame() {{
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (showBoxes) {{
      const frameIdx  = String(Math.floor(vid.currentTime * FPS));
      const frameData = BOXES[frameIdx] || [];
      const scaleX    = canvas.width  / VID_W;
      const scaleY    = canvas.height / VID_H;

      ctx.strokeStyle = '#00ff44';
      ctx.lineWidth   = 2;
      ctx.fillStyle   = '#00ff44';
      ctx.font        = 'bold 13px sans-serif';

      for (const [x1, y1, x2, y2, conf] of frameData) {{
        const sx = x1 * scaleX, sy = y1 * scaleY;
        const sw = (x2 - x1) * scaleX, sh = (y2 - y1) * scaleY;
        ctx.strokeRect(sx, sy, sw, sh);
        ctx.fillText(Math.round(conf * 100) + '%', sx + 2, sy - 4);
      }}
    }}
    requestAnimationFrame(drawFrame);
  }}

  function toggleBoxes() {{
    showBoxes = !showBoxes;
    const btn = document.getElementById('toggleBtn');
    btn.textContent = 'Boxes: ' + (showBoxes ? 'ON' : 'OFF');
    btn.classList.toggle('off', !showBoxes);
  }}

  vid.addEventListener('loadedmetadata', () => {{ resizeCanvas(); drawFrame(); }});
  window.addEventListener('resize', resizeCanvas);
</script>
"""


# ─── UI ──────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if st.button("Run Detection", type="primary"):
        (clean_frames, annotated_frames, counts,
         frame_boxes, frame_w, frame_h, fps, elapsed) = process_video(tmp_path, load_model())
        os.unlink(tmp_path)

        st.session_state.clean_frames     = clean_frames
        st.session_state.annotated_frames = annotated_frames
        st.session_state.counts           = counts
        st.session_state.frame_boxes      = frame_boxes
        st.session_state.frame_w          = frame_w
        st.session_state.frame_h          = frame_h
        st.session_state.fps              = fps
        st.session_state.elapsed          = elapsed
        st.session_state.pop("video_url", None)   # clear old player

if "annotated_frames" in st.session_state:
    annotated_frames = st.session_state.annotated_frames
    clean_frames     = st.session_state.clean_frames
    counts           = st.session_state.counts
    frame_boxes      = st.session_state.frame_boxes
    frame_w          = st.session_state.frame_w
    frame_h          = st.session_state.frame_h
    fps              = st.session_state.fps
    elapsed          = st.session_state.elapsed

    st.success(f"Processed {len(annotated_frames)} frames in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total frames",     len(annotated_frames))
    col2.metric("Avg people/frame", f"{np.mean(counts):.1f}")
    col3.metric("Peak people",      max(counts) if counts else 0)

    st.divider()

    tab_scrubber, tab_play = st.tabs(["Frame Scrubber", "Play Video"])

    with tab_scrubber:
        frame_idx = st.slider("Scrub through frames", 0, len(annotated_frames) - 1, 0)
        st.image(annotated_frames[frame_idx],
                 caption=f"Frame {frame_idx} — {counts[frame_idx]} people",
                 width='stretch')

    with tab_play:
        if st.button("Render & Play Video"):
            video_path = render_video(clean_frames, fps, label="Rendering video")
            if video_path is not None:
                url = _start_video_server(video_path)
                st.session_state.video_url = url

        if "video_url" in st.session_state:
            html = build_player_html(
                st.session_state.video_url,
                frame_boxes, frame_w, frame_h, fps
            )
            # Estimate displayed video height: wide layout content area ≈ 1400px wide
            video_display_h = int(1400 * frame_h / frame_w)
            iframe_h = video_display_h + 80   # +80 for video controls row + button row
            components.html(html, height=iframe_h, scrolling=False)

else:
    st.info("Upload a video to get started.")
