import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from scipy.ndimage import gaussian_filter1d

_BASE = Path(__file__).resolve().parent.parent


def _load(module_name: str, file_path: Path):
    """Load a module from an explicit file path and cache it in sys.modules."""
    spec   = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)       # type: ignore[arg-type]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)                      # type: ignore[union-attr]
    return module


_sc = _load("smartCroppingVideos",
            _BASE / "cropping" / "smartCroppingVideos.py")

# Re-export names so the IDE and callers can see them
cropped_width:  int   = _sc.cropped_width
cropped_height: int   = _sc.cropped_height

GAUSSIAN_SIGMA:        int   = 15
BALL_MIN_AREA_FRAC:    float = 0.0001
BALL_MAX_AREA_FRAC:    float = 0.02
OUTLIER_MAD_THRESHOLD: float = 3.0


# ─── Shared crop helpers ──────────────────────────────────────────────────────

def _choose_best_ball(boxes, ball_class_id: int):
    """Return (box_xyxy, confidence) for the highest-conf sports ball, or None."""
    if boxes is None or len(boxes) == 0:
        return None
    confs        = boxes.conf.cpu().numpy()
    classes      = boxes.cls.cpu().numpy().astype(int)
    ball_indices = np.where(classes == ball_class_id)[0]
    if ball_indices.size == 0:
        return None
    best = ball_indices[np.argmax(confs[ball_indices])]
    return boxes.xyxy[best].cpu().numpy(), float(confs[best])


def _is_valid_ball_size(box_xyxy, frame_w: int, frame_h: int) -> bool:
    """Return True if the bounding box area is within a plausible basketball range."""
    x1, y1, x2, y2 = box_xyxy
    frac = (x2 - x1) * (y2 - y1) / (frame_w * frame_h)
    return BALL_MIN_AREA_FRAC <= frac <= BALL_MAX_AREA_FRAC


def _reject_spatial_outliers(known_indices: list, known_x1s: list,
                              known_cxs: list) -> tuple:
    """Remove detections whose cx is a MAD-sigma outlier across the full clip."""
    if len(known_cxs) < 4:
        return known_indices, known_x1s
    cx_arr    = np.array(known_cxs, dtype=np.float64)
    median_cx = np.median(cx_arr)
    mad       = np.median(np.abs(cx_arr - median_cx))
    if mad < 1e-6:
        return known_indices, known_x1s
    z    = np.abs(cx_arr - median_cx) / (mad / 0.6745)
    mask = z <= OUTLIER_MAD_THRESHOLD
    return (
        [known_indices[i] for i in range(len(known_indices)) if mask[i]],
        [known_x1s[i]     for i in range(len(known_x1s))     if mask[i]],
    )


# ─── Player detection ─────────────────────────────────────────────────────────

def draw_tracked_boxes(frame, boxes, track_ids):
    for box, tid in zip(boxes, track_ids):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {int(tid)}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
    frame_boxes      = {}     # {frame_index: [[x1,y1,x2,y2,track_id], ...]}
    progress         = st.progress(0, text="Running player detection...")
    start            = time.time()

    for i, result in enumerate(model.track(
            source=video_path, classes=[0], stream=True,
            imgsz=416, conf=0.35, iou=0.8,
            persist=True, tracker="bytetrack.yaml", verbose=False)):
        frame_rgb  = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
        boxes      = result.boxes.xyxy.cpu().numpy() if result.boxes else np.empty((0, 4))
        track_ids  = (result.boxes.id.cpu().numpy()
                      if result.boxes is not None and result.boxes.id is not None
                      else np.zeros(len(boxes)))

        clean_frames.append(frame_rgb.copy())
        annotated_frames.append(draw_tracked_boxes(frame_rgb.copy(), boxes, track_ids))
        frame_counts.append(int(len(boxes)))

        if len(boxes):
            frame_boxes[str(i)] = [
                [float(x1), float(y1), float(x2), float(y2), int(tid)]
                for (x1, y1, x2, y2), tid in zip(boxes, track_ids)
            ]

        if total_frames > 0:
            progress.progress(min((i + 1) / total_frames, 1.0),
                              text=f"Frame {i+1}/{total_frames} — {len(boxes)} people detected")

    progress.empty()
    return (clean_frames, annotated_frames, frame_counts,
            frame_boxes, frame_w, frame_h, fps, time.time() - start)


# ─── Ball detection ───────────────────────────────────────────────────────────

def process_ball_video(video_path, model):
    ball_class_id = [k for k, v in model.names.items() if v == "sports ball"][0]

    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    clean_frames     = []
    frame_ball_boxes = {}   # {str(i): [x1, y1, x2, y2, conf]}
    ball_counts      = []
    known_indices    = []
    known_cxs        = []
    known_cys        = []

    progress = st.progress(0, text="Running ball detection...")
    start    = time.time()

    # Pass 1 — detect ball in every frame
    for i, result in enumerate(model.predict(
            source=video_path, classes=[ball_class_id], stream=True,
            save=False, conf=0.20, iou=0.4, imgsz=768)):
        frame_rgb = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
        clean_frames.append(frame_rgb.copy())

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

                frame_ball_boxes[str(i)] = [x1, y1, x2, y2, conf]
                known_indices.append(i)
                known_cxs.append(cx)
                known_cys.append(cy)
                have_det = True

        ball_counts.append(1 if have_det else 0)

        if total_frames > 0:
            progress.progress(min((i + 1) / total_frames, 1.0),
                              text=f"Frame {i+1}/{total_frames} — {'ball detected' if have_det else 'no ball'}")

    progress.empty()

    # Pass 1.5 — gap-fill then Gaussian smooth the full cx/cy trajectory
    n = len(clean_frames)
    frame_gaussian = {}
    if known_indices and n > 0:
        all_indices = np.arange(n)
        smooth_cx = gaussian_filter1d(
            np.interp(all_indices, known_indices, known_cxs),
            sigma=GAUSSIAN_SIGMA, mode="nearest",
        )
        smooth_cy = gaussian_filter1d(
            np.interp(all_indices, known_indices, known_cys),
            sigma=GAUSSIAN_SIGMA, mode="nearest",
        )
        for i in range(n):
            frame_gaussian[str(i)] = [float(smooth_cx[i]), float(smooth_cy[i])]

    return (clean_frames, frame_ball_boxes, frame_gaussian,
            ball_counts, frame_w, frame_h, fps, time.time() - start)


# ─── Smart crop ───────────────────────────────────────────────────────────────

def process_smart_crop_video(video_path, model):
    ball_class_id = [k for k, v in model.names.items() if v == "sports ball"][0]

    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    target_w = min(int(frame_h * 9 / 16), frame_w)
    max_x1   = frame_w - target_w

    clean_frames  = []
    known_indices = []
    known_x1s     = []
    known_cxs     = []
    known_cys     = []
    idx_to_box    = {}   # frame index → (box_xyxy, conf)

    progress = st.progress(0, text="Running smart crop...")
    start    = time.time()

    # Pass 1 — detect ball in every frame, collect crop positions
    for i, result in enumerate(model.predict(
            source=video_path, classes=[ball_class_id], stream=True,
            save=False, conf=0.20, iou=0.4, imgsz=768)):
        clean_frames.append(cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB))

        detection = _choose_best_ball(result.boxes, ball_class_id)
        if detection is not None:
            box_xyxy, conf = detection
            if _is_valid_ball_size(box_xyxy, frame_w, frame_h):
                cx = (box_xyxy[0] + box_xyxy[2]) / 2.0
                cy = (box_xyxy[1] + box_xyxy[3]) / 2.0
                x1 = float(np.clip(cx - target_w / 2.0, 0, max_x1))
                known_indices.append(i)
                known_x1s.append(x1)
                known_cxs.append(cx)
                known_cys.append(cy)
                idx_to_box[i] = (box_xyxy, conf)

        if total_frames > 0:
            progress.progress(min((i + 1) / total_frames, 1.0),
                              text=f"Frame {i+1}/{total_frames}")

    progress.empty()

    n = len(clean_frames)
    if n == 0:
        return None

    # Outlier rejection — save cx/cy lookup first so we can re-align after
    _cx_by_idx = dict(zip(known_indices, known_cxs))
    _cy_by_idx = dict(zip(known_indices, known_cys))

    known_indices, known_x1s = _reject_spatial_outliers(known_indices, known_x1s, known_cxs)
    frame_boxes = {idx: idx_to_box[idx] for idx in known_indices if idx in idx_to_box}

    known_cxs = [_cx_by_idx[i] for i in known_indices]
    known_cys = [_cy_by_idx[i] for i in known_indices]

    # Gap-fill via linear interpolation
    all_idx = np.arange(n)
    if len(known_indices) == 0:
        raw_x1s = np.full(n, max_x1 / 2.0)
        raw_cxs = np.full(n, frame_w  / 2.0)
        raw_cys = np.full(n, frame_h  / 2.0)
    else:
        raw_x1s = np.interp(all_idx, known_indices, known_x1s)
        raw_cxs = np.interp(all_idx, known_indices, known_cxs)
        raw_cys = np.interp(all_idx, known_indices, known_cys)

    # Pass 1.5 — Gaussian smooth the full trajectory (symmetric, no causal lag)
    smoothed_x1s = np.clip(
        gaussian_filter1d(raw_x1s, sigma=GAUSSIAN_SIGMA, mode="nearest"),
        0, max_x1,
    )
    smoothed_cxs = gaussian_filter1d(raw_cxs, sigma=GAUSSIAN_SIGMA, mode="nearest")
    smoothed_cys = gaussian_filter1d(raw_cys, sigma=GAUSSIAN_SIGMA, mode="nearest")

    # Pass 2 — crop each frame using smoothed x1 positions, build overlay dicts
    sx = cropped_width  / target_w
    sy = cropped_height / frame_h

    cropped_frames        = []
    frame_ball_boxes_crop = {}
    frame_pred_crop       = {}

    for i, x1 in enumerate(smoothed_x1s):
        x1_int  = int(round(x1))
        frame   = clean_frames[i]   # already RGB
        cropped = frame[:, x1_int: x1_int + target_w]
        cropped_frames.append(cv2.resize(cropped, (cropped_width, cropped_height)))

        if i in frame_boxes:
            bx1, by1, bx2, by2 = frame_boxes[i][0]
            conf = frame_boxes[i][1]
            frame_ball_boxes_crop[str(i)] = [
                float((bx1 - x1_int) * sx), float(by1 * sy),
                float((bx2 - x1_int) * sx), float(by2 * sy),
                conf,
            ]

        # Smoothed ball position mapped into cropped-frame coordinates
        frame_pred_crop[str(i)] = [
            float((smoothed_cxs[i] - x1_int) * sx),
            float(smoothed_cys[i] * sy),
        ]

    return (cropped_frames, frame_ball_boxes_crop, frame_pred_crop,
            fps, time.time() - start)