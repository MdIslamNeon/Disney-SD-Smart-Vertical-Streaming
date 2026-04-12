import os
#? Note that these env variables are for linux. The current terminal outputs OpenH264 library error but can be ignored for now
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # AV_LOG_QUIET

import atexit
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

st.set_page_config(page_title="Smart Vertical Basketball Streaming", layout="wide", initial_sidebar_state="collapsed")
st.title("Smart Vertical Basketball Streaming")


# ─── Local video server ───────────────────────────────────────────────────────
# Serves the rendered video file over HTTP so the browser can fetch it directly,
# avoiding the Streamlit websocket size limit.
#
# Servers are keyed by label ("player" / "ball") so both can coexist without
# one clobbering the other's temp file.

_active_servers: dict[str, dict] = {}  # label -> {"server": HTTPServer, "path": str}
_rendered_paths: list[str] = []        # all temp files created this session

def _cleanup_temp_files() -> None:
    """Delete all rendered temp files — registered with atexit."""
    for p in _rendered_paths:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass

atexit.register(_cleanup_temp_files)

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def _start_video_server(video_path: str, label: str = "player") -> str:
    """Shut down any previous server for this label, then serve video_path on a new port."""
    global _active_servers
    if label in _active_servers:
        _active_servers[label]["server"].shutdown()
        _active_servers.pop(label)

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
    _active_servers[label] = {"server": server, "path": video_path}
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


def kalman_filter():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix  = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                      [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.06
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.3
    kf.errorCovPost        = np.eye(4, dtype=np.float32)
    return kf


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
    progress         = st.progress(0, text="Running player detection...")
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


def process_ball_video(video_path, model):
    ball_class_id = [k for k, v in model.names.items() if v == "sports ball"][0]

    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    kf             = kalman_filter()
    kf_initialized = False

    clean_frames     = []
    frame_ball_boxes = {}   # {str(i): [x1, y1, x2, y2, conf]}
    frame_kalman     = {}   # {str(i): [px, py]}
    ball_counts      = []

    progress = st.progress(0, text="Running ball detection...")
    start    = time.time()

    for i, result in enumerate(model.predict(
            source=video_path, classes=[ball_class_id], stream=True,
            save=False, conf=0.20, iou=0.4, imgsz=768)):
        frame_rgb = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
        clean_frames.append(frame_rgb.copy())

        # Always predict first — matches ball_detector_test.py logic
        pred = kf.predict()
        px, py = float(pred[0]), float(pred[1])

        have_det = False
        if result.boxes is not None and len(result.boxes) > 0:
            confs   = result.boxes.conf.cpu().numpy()   # type: ignore[union-attr]
            classes = result.boxes.cls.cpu().numpy()    # type: ignore[union-attr]

            idx = int(np.argmax(confs))
            if int(classes[idx]) == ball_class_id:
                xyxy = result.boxes.xyxy[idx].cpu().numpy()  # type: ignore[union-attr]
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                conf = float(confs[idx])
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
                if not kf_initialized:
                    kf.statePre[:2]  = measurement
                    kf.statePost[:2] = measurement
                    kf_initialized   = True

                kf.correct(measurement)

                # Hard-reset when YOLO is very confident
                if conf > 0.6:
                    kf.statePre[:2]  = measurement
                    kf.statePost[:2] = measurement

                frame_ball_boxes[str(i)] = [x1, y1, x2, y2, conf]
                have_det = True

        ball_counts.append(1 if have_det else 0)

        # Store the pre-correction Kalman prediction for this frame
        if kf_initialized:
            frame_kalman[str(i)] = [px, py]

        if total_frames > 0:
            progress.progress(min((i + 1) / total_frames, 1.0),
                              text=f"Frame {i+1}/{total_frames} — {'ball detected' if have_det else 'no ball'}")

    progress.empty()
    return (clean_frames, frame_ball_boxes, frame_kalman,
            ball_counts, frame_w, frame_h, fps, time.time() - start)


def render_video(frames, fps, label="Rendering video") -> str | None:
    """Write RGB frames to a *persistent* temp file (H.264). Returns the file path."""
    h, w = frames[0].shape[:2]

    # delete=False so the file stays alive for the HTTP server to serve
    out_tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = out_tmp.name
    out_tmp.close()
    _rendered_paths.append(out_path)

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
  <button id="toggleBtn" onclick="toggleBoxes()">Player Detection Boxes: ON</button>
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
    btn.textContent = 'Player Detection Boxes: ' + (showBoxes ? 'ON' : 'OFF');
    btn.classList.toggle('off', !showBoxes);
  }}

  vid.addEventListener('loadedmetadata', () => {{ resizeCanvas(); drawFrame(); }});
  window.addEventListener('resize', resizeCanvas);
</script>
"""


def build_ball_html(video_url: str, frame_ball_boxes: dict, frame_kalman: dict,
                    frame_w: int, frame_h: int, fps: float) -> str:
    ball_json   = json.dumps(frame_ball_boxes)
    kalman_json = json.dumps(frame_kalman)
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
    height: auto;
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
    gap: 8px;
    padding: 4px 8px;
    background: #111;
  }}
  .toggleBtn {{
    padding: 5px 18px;
    font-size: 13px;
    font-weight: bold;
    border-radius: 6px;
    cursor: pointer;
  }}
  #ballBoxBtn {{
    border: 2px solid #ffff00;
    background: #ffff00;
    color: #fff;
  }}
  #ballBoxBtn.off {{
    background: #333;
    border-color: #555;
    color: #aaa;
  }}
  #kalmanBtn {{
    border: 2px solid #ff2222;
    background: #ff2222;
    color: #fff;
  }}
  #kalmanBtn.off {{
    background: #333;
    border-color: #555;
    color: #aaa;
  }}
</style>

<div id="controls">
  <button id="ballBoxBtn" class="toggleBtn" onclick="toggleBallBox()">Ball Box: ON</button>
  <button id="kalmanBtn"  class="toggleBtn" onclick="toggleKalman()">Kalman Prediction: ON</button>
</div>
<div id="player-wrap">
  <video id="vid" controls>
    <source src="{video_url}" type="video/mp4">
  </video>
  <canvas id="overlay"></canvas>
</div>

<script>
  const BALL_BOXES = {ball_json};
  const KALMAN     = {kalman_json};
  const FPS   = {fps};
  const VID_W = {frame_w};
  const VID_H = {frame_h};
  let showBallBox = true;
  let showKalman  = true;

  const vid    = document.getElementById('vid');
  const canvas = document.getElementById('overlay');
  const ctx    = canvas.getContext('2d');

  function resizeCanvas() {{
    canvas.width        = vid.clientWidth;
    canvas.height       = vid.clientHeight;
    canvas.style.width  = vid.clientWidth  + 'px';
    canvas.style.height = vid.clientHeight + 'px';
  }}

  function drawFrame() {{
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const frameIdx = String(Math.floor(vid.currentTime * FPS));
    const scaleX   = canvas.width  / VID_W;
    const scaleY   = canvas.height / VID_H;

    if (showBallBox && BALL_BOXES[frameIdx]) {{
      const [x1, y1, x2, y2, conf] = BALL_BOXES[frameIdx];
      ctx.strokeStyle = '#ffff00';
      ctx.lineWidth   = 3;
      ctx.strokeRect(x1*scaleX, y1*scaleY, (x2-x1)*scaleX, (y2-y1)*scaleY);
      ctx.fillStyle = '#ffff00';
      ctx.font      = 'bold 13px sans-serif';
      ctx.fillText(Math.round(conf*100) + '%', x1*scaleX + 2, y1*scaleY - 4);
    }}

    if (showKalman && KALMAN[frameIdx]) {{
      const [px, py] = KALMAN[frameIdx];
      ctx.strokeStyle = '#ff2222';
      ctx.lineWidth   = 2;
      ctx.beginPath();
      ctx.arc(px*scaleX, py*scaleY, 12, 0, 2*Math.PI);
      ctx.stroke();
    }}

    requestAnimationFrame(drawFrame);
  }}

  function toggleBallBox() {{
    showBallBox = !showBallBox;
    const btn = document.getElementById('ballBoxBtn');
    btn.textContent = 'Ball Box: ' + (showBallBox ? 'ON' : 'OFF');
    btn.classList.toggle('off', !showBallBox);
  }}

  function toggleKalman() {{
    showKalman = !showKalman;
    const btn = document.getElementById('kalmanBtn');
    btn.textContent = 'Kalman Prediction: ' + (showKalman ? 'ON' : 'OFF');
    btn.classList.toggle('off', !showKalman);
  }}

  vid.addEventListener('loadedmetadata', () => {{ resizeCanvas(); drawFrame(); }});
  window.addEventListener('resize', resizeCanvas);
</script>
"""


# ─── UI ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Menu")
    if st.button("Quit App", type="primary", use_container_width=True):
        components.html(
            "<script>window.top.close(); window.top.location.href='about:blank';</script>",
            height=0,
        )
        time.sleep(0.3)
        os._exit(0)

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Keep the tmp file alive across reruns so both detection buttons can use it
    if st.session_state.get("uploaded_name") != uploaded_file.name:
        # Clean up previous upload temp
        old_tmp = st.session_state.get("tmp_path")
        if old_tmp and os.path.exists(old_tmp):
            os.unlink(old_tmp)
        # Clean up any previously rendered output files
        for key in ("video_path", "ball_video_path"):
            old_rendered = st.session_state.pop(key, None)
            if old_rendered and os.path.exists(old_rendered):
                os.unlink(old_rendered)
        st.session_state.pop("video_url", None)
        st.session_state.pop("ball_video_url", None)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            st.session_state.tmp_path      = tmp.name
            st.session_state.uploaded_name = uploaded_file.name

    tmp_path = st.session_state.tmp_path

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Player Detection", type="primary", use_container_width=True):
            (clean_frames, annotated_frames, counts,
             frame_boxes, frame_w, frame_h, fps, elapsed) = process_video(tmp_path, load_model())

            st.session_state.clean_frames     = clean_frames
            st.session_state.annotated_frames = annotated_frames
            st.session_state.counts           = counts
            st.session_state.frame_boxes      = frame_boxes
            st.session_state.frame_w          = frame_w
            st.session_state.frame_h          = frame_h
            st.session_state.fps              = fps
            st.session_state.elapsed          = elapsed
            # Invalidate cached render so the new detection results are used
            st.session_state.pop("video_path", None)
            st.session_state.pop("video_url", None)

    with col2:
        if st.button("Run Ball Detection", type="primary", use_container_width=True):
            (ball_clean_frames, frame_ball_boxes, frame_kalman,
             ball_counts, ball_frame_w, ball_frame_h, ball_fps, ball_elapsed) = process_ball_video(tmp_path, load_model())

            st.session_state.ball_clean_frames  = ball_clean_frames
            st.session_state.frame_ball_boxes   = frame_ball_boxes
            st.session_state.frame_kalman       = frame_kalman
            st.session_state.ball_counts        = ball_counts
            st.session_state.ball_frame_w       = ball_frame_w
            st.session_state.ball_frame_h       = ball_frame_h
            st.session_state.ball_fps           = ball_fps
            st.session_state.ball_elapsed       = ball_elapsed
            # Invalidate cached render so the new detection results are used
            st.session_state.pop("ball_video_path", None)
            st.session_state.pop("ball_video_url", None)

# ─── Results ──────────────────────────────────────────────────────────────────

has_player = "annotated_frames" in st.session_state
has_ball   = "frame_ball_boxes" in st.session_state

if has_player or has_ball:
    if has_player:
        annotated_frames = st.session_state.annotated_frames
        clean_frames     = st.session_state.clean_frames
        counts           = st.session_state.counts
        frame_boxes      = st.session_state.frame_boxes
        frame_w          = st.session_state.frame_w
        frame_h          = st.session_state.frame_h
        fps              = st.session_state.fps
        elapsed          = st.session_state.elapsed

        st.subheader("Player Detection")
        st.success(f"Processed {len(annotated_frames)} frames in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total frames",     len(annotated_frames))
        col2.metric("Avg people/frame", f"{np.mean(counts):.1f}")
        col3.metric("Peak people",      max(counts) if counts else 0)

    if has_ball:
        frame_ball_boxes  = st.session_state.frame_ball_boxes
        frame_kalman      = st.session_state.frame_kalman
        ball_counts       = st.session_state.ball_counts
        ball_frame_w      = st.session_state.ball_frame_w
        ball_frame_h      = st.session_state.ball_frame_h
        ball_fps          = st.session_state.ball_fps
        ball_elapsed      = st.session_state.ball_elapsed
        ball_clean_frames = st.session_state.ball_clean_frames

        total_ball_frames = len(ball_counts)
        frames_with_ball  = sum(ball_counts)
        detection_rate    = frames_with_ball / total_ball_frames * 100 if total_ball_frames else 0

        st.subheader("Ball Detection")
        st.success(f"Processed {total_ball_frames} frames in {ball_elapsed:.1f}s ({ball_elapsed/60:.1f} min)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total frames",     total_ball_frames)
        col2.metric("Frames with ball", frames_with_ball)
        col3.metric("Detection rate",   f"{detection_rate:.1f}%")

    st.divider()

    # Build tab list dynamically based on what results are available
    tab_names = []
    if has_player:
        tab_names += ["Frame Scrubber", "Player Play Video"]
    if has_ball:
        tab_names += ["Ball Play Video"]

    tabs     = st.tabs(tab_names)
    tab_idx  = 0

    if has_player:
        with tabs[tab_idx]:   # Frame Scrubber
            frame_idx = st.slider("Scrub through frames", 0, len(annotated_frames) - 1, 0)

            # Overlay toggles
            chk_cols = st.columns(3 if has_ball else 1)
            show_player_boxes = chk_cols[0].checkbox("Player Boxes", value=True)
            if has_ball:
                show_ball_box = chk_cols[1].checkbox("Ball Box", value=True)
                show_kalman   = chk_cols[2].checkbox("Kalman Prediction", value=True)

            # Start from clean or annotated base depending on player toggle
            display_frame = (annotated_frames[frame_idx] if show_player_boxes
                             else clean_frames[frame_idx]).copy()

            # Draw ball box (orange) if ball detection ran and toggle is on
            if has_ball and show_ball_box and str(frame_idx) in frame_ball_boxes:
                x1, y1, x2, y2, conf = frame_ball_boxes[str(frame_idx)]
                cv2.rectangle(display_frame,
                              (int(x1), int(y1)), (int(x2), int(y2)),
                              (255, 255, 0), 3)   # yellow in RGB
                cv2.putText(display_frame, f"{conf:.0%}",
                            (int(x1), max(0, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            # Draw Kalman circle (red) if ball detection ran and toggle is on
            if has_ball and show_kalman and str(frame_idx) in frame_kalman:
                px, py = frame_kalman[str(frame_idx)]
                cv2.circle(display_frame, (int(px), int(py)), 12, (255, 34, 34), 2)  # red in RGB

            ball_status = ""
            if has_ball:
                ball_status = (" — ball detected" if str(frame_idx) in frame_ball_boxes
                               else " — no ball")
            st.image(display_frame,
                     caption=f"Frame {frame_idx} — {counts[frame_idx]} people{ball_status}",
                     width='stretch')
        tab_idx += 1

        with tabs[tab_idx]:   # Player Play Video
            if st.button("Render & Play Player Video"):
                existing = st.session_state.get("video_path")
                if existing and os.path.exists(existing):
                    video_path = existing
                else:
                    video_path = render_video(clean_frames, fps, label="Rendering player video")
                    if video_path is not None:
                        st.session_state.video_path = video_path
                if video_path is not None:
                    url = _start_video_server(video_path, label="player")
                    st.session_state.video_url = url

            if "video_url" in st.session_state:
                html = build_player_html(
                    st.session_state.video_url,
                    frame_boxes, frame_w, frame_h, fps
                )
                video_display_h = int(1400 * frame_h / frame_w)
                components.html(html, height=video_display_h + 80, scrolling=False)
        tab_idx += 1

    if has_ball:
        with tabs[tab_idx]:   # Ball Play Video
            if st.button("Render & Play Ball Video"):
                existing = st.session_state.get("ball_video_path")
                if existing and os.path.exists(existing):
                    ball_video_path = existing
                else:
                    ball_video_path = render_video(ball_clean_frames, ball_fps, label="Rendering ball video")
                    if ball_video_path is not None:
                        st.session_state.ball_video_path = ball_video_path
                if ball_video_path is not None:
                    url = _start_video_server(ball_video_path, label="ball")
                    st.session_state.ball_video_url = url

            if "ball_video_url" in st.session_state:
                html = build_ball_html(
                    st.session_state.ball_video_url,
                    frame_ball_boxes, frame_kalman,
                    ball_frame_w, ball_frame_h, ball_fps
                )
                video_display_h = int(1400 * ball_frame_h / ball_frame_w)
                components.html(html, height=video_display_h + 80, scrolling=False)

else:
    st.info("Upload a video to get started.")
