import os
#? Note that these env variables are for linux. The current terminal outputs OpenH264 library error but can be ignored for now
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # AV_LOG_QUIET

import tempfile
import time

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from ultralytics import YOLO

#? Pylance throws error but still works
from detection import (process_video, process_ball_video, process_smart_crop_video, cropped_width, cropped_height) # type: ignore
from html_builders import build_player_html, build_ball_html, build_smart_crop_html, build_final_product_html
from video_utils import render_video, _start_video_server

st.set_page_config(page_title="Smart Vertical Basketball Streaming", layout="wide", initial_sidebar_state="collapsed")

title_col, quit_col = st.columns([8, 1])
title_col.title("Smart Vertical Basketball Streaming")
if quit_col.button("Quit App", type="primary", use_container_width=True):
    components.html(
        "<script>window.top.close(); window.top.location.href='about:blank';</script>",
        height=0,
    )
    time.sleep(0.3)
    os._exit(0)


# ─── Model ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return YOLO("../models/yolov8m.pt")


# ─── Upload ───────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Keep the tmp file alive across reruns so all three buttons can use it
    if st.session_state.get("uploaded_name") != uploaded_file.name:
        old_tmp = st.session_state.get("tmp_path")
        if old_tmp and os.path.exists(old_tmp):
            os.unlink(old_tmp)
        for key in ("video_path", "ball_video_path", "smart_crop_video_path", "final_product_video_path"):
            old_rendered = st.session_state.pop(key, None)
            if old_rendered and os.path.exists(old_rendered):
                os.unlink(old_rendered)
        st.session_state.pop("video_url", None)
        st.session_state.pop("ball_video_url", None)
        st.session_state.pop("smart_crop_video_url", None)
        st.session_state.pop("final_product_video_url", None)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            st.session_state.tmp_path      = tmp.name
            st.session_state.uploaded_name = uploaded_file.name

    tmp_path = st.session_state.tmp_path

    if st.button("Automate Video Clip", type="primary", use_container_width=True):
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
        st.session_state.pop("video_path", None)
        st.session_state.pop("video_url", None)

        (ball_clean_frames, frame_ball_boxes, frame_gaussian,
         ball_counts, ball_frame_w, ball_frame_h, ball_fps, ball_elapsed) = process_ball_video(tmp_path, load_model())

        st.session_state.ball_clean_frames  = ball_clean_frames
        st.session_state.frame_ball_boxes   = frame_ball_boxes
        st.session_state.frame_gaussian     = frame_gaussian
        st.session_state.ball_counts        = ball_counts
        st.session_state.ball_frame_w       = ball_frame_w
        st.session_state.ball_frame_h       = ball_frame_h
        st.session_state.ball_fps           = ball_fps
        st.session_state.ball_elapsed       = ball_elapsed
        st.session_state.pop("ball_video_path", None)
        st.session_state.pop("ball_video_url", None)

        result = process_smart_crop_video(tmp_path, load_model())
        if result is not None:
            (sc_frames, sc_ball_boxes, sc_pred,
             sc_fps, sc_elapsed,
             sc_smoothed_x1s, sc_sx, sc_sy) = result

            st.session_state.sc_frames       = sc_frames
            st.session_state.sc_ball_boxes   = sc_ball_boxes
            st.session_state.sc_pred         = sc_pred
            st.session_state.sc_fps          = sc_fps
            st.session_state.sc_elapsed      = sc_elapsed
            st.session_state.sc_smoothed_x1s = sc_smoothed_x1s
            st.session_state.sc_sx           = sc_sx
            st.session_state.sc_sy           = sc_sy
            st.session_state.pop("smart_crop_video_path", None)
            st.session_state.pop("smart_crop_video_url", None)


# ─── Results ──────────────────────────────────────────────────────────────────

has_player     = "annotated_frames" in st.session_state
has_ball       = "frame_ball_boxes" in st.session_state
has_smart_crop    = "sc_frames"        in st.session_state
has_final_product = has_smart_crop and has_player

if has_player or has_ball or has_smart_crop:
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
        frame_gaussian    = st.session_state.frame_gaussian
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

    if has_smart_crop:
        sc_elapsed = st.session_state.sc_elapsed
        sc_frames  = st.session_state.sc_frames

        st.subheader("Smart Crop")
        st.success(f"Processed {len(sc_frames)} frames in {sc_elapsed:.1f}s ({sc_elapsed/60:.1f} min)")
        col1, col2 = st.columns(2)
        col1.metric("Total frames", len(sc_frames))
        col2.metric("Output size",  f"{cropped_width}×{cropped_height}")

    st.divider()

    # Build tab list dynamically based on what results are available
    tab_names = []
    if has_player:
        tab_names += ["Frame Scrubber", "Player Play Video"]
    if has_ball:
        tab_names += ["Ball Play Video"]
    if has_smart_crop:
        tab_names += ["Smart Crop Video"]
    if has_final_product:
        tab_names += ["Final Product"]

    tabs    = st.tabs(tab_names)
    tab_idx = 0

    if has_player:
        with tabs[tab_idx]:   # Frame Scrubber
            frame_idx = st.slider("Scrub through frames", 0, len(annotated_frames) - 1, 0)

            chk_cols = st.columns(3 if has_ball else 1)
            show_player_boxes = chk_cols[0].checkbox("Player Boxes", value=True)
            if has_ball:
                show_ball_box = chk_cols[1].checkbox("Ball Box", value=True)
                show_gaussian = chk_cols[2].checkbox("Gaussian Smoothing", value=True)

            display_frame = (annotated_frames[frame_idx] if show_player_boxes
                             else clean_frames[frame_idx]).copy()

            if has_ball and show_ball_box and str(frame_idx) in frame_ball_boxes:
                x1, y1, x2, y2, conf = frame_ball_boxes[str(frame_idx)]
                cv2.rectangle(display_frame,
                              (int(x1), int(y1)), (int(x2), int(y2)),
                              (255, 255, 0), 3)
                cv2.putText(display_frame, f"{conf:.0%}",
                            (int(x1), max(0, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            if has_ball and show_gaussian and str(frame_idx) in frame_gaussian:
                px, py = frame_gaussian[str(frame_idx)]
                cv2.circle(display_frame, (int(px), int(py)), 12, (255, 34, 34), 2)

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
                    frame_ball_boxes, frame_gaussian,
                    ball_frame_w, ball_frame_h, ball_fps
                )
                video_display_h = int(1400 * ball_frame_h / ball_frame_w)
                components.html(html, height=video_display_h + 80, scrolling=False)
        tab_idx += 1

    if has_smart_crop:
        with tabs[tab_idx]:   # Smart Crop Video
            sc_frames     = st.session_state.sc_frames
            sc_ball_boxes = st.session_state.sc_ball_boxes
            sc_pred       = st.session_state.sc_pred
            sc_fps        = st.session_state.sc_fps

            if st.button("Render & Play Smart Crop Video"):
                existing = st.session_state.get("smart_crop_video_path")
                if existing and os.path.exists(existing):
                    sc_video_path = existing
                else:
                    sc_video_path = render_video(sc_frames, sc_fps,
                                                 label="Rendering smart crop video")
                    if sc_video_path is not None:
                        st.session_state.smart_crop_video_path = sc_video_path
                if sc_video_path is not None:
                    url = _start_video_server(sc_video_path, label="smart_crop")
                    st.session_state.smart_crop_video_url = url

            if "smart_crop_video_url" in st.session_state:
                html = build_smart_crop_html(
                    st.session_state.smart_crop_video_url,
                    sc_ball_boxes, sc_pred, sc_fps
                )
                components.html(html, height=850, scrolling=False)
        tab_idx += 1

    if has_final_product:
        with tabs[tab_idx]:   # Final Product
            sc_ball_boxes    = st.session_state.sc_ball_boxes
            sc_fps           = st.session_state.sc_fps
            sc_smoothed_x1s  = st.session_state.sc_smoothed_x1s
            sc_sx            = st.session_state.sc_sx
            sc_sy            = st.session_state.sc_sy

            if st.button("Render & Play Final Product"):
                existing = st.session_state.get("final_product_video_path")
                if existing and os.path.exists(existing):
                    fp_video_path = existing
                else:
                    fp_video_path = render_video(st.session_state.sc_frames, sc_fps,
                                                 label="Rendering final product video")
                    if fp_video_path is not None:
                        st.session_state.final_product_video_path = fp_video_path
                if fp_video_path is not None:
                    url = _start_video_server(fp_video_path, label="final_product")
                    st.session_state.final_product_video_url = url

            if "final_product_video_url" in st.session_state:
                html = build_final_product_html(
                    st.session_state.final_product_video_url,
                    frame_boxes, sc_ball_boxes, st.session_state.sc_pred,
                    sc_smoothed_x1s, sc_sx, sc_sy, sc_fps
                )
                components.html(html, height=850, scrolling=False)