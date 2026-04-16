import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

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
_bd = _load("ball_detection",
            _BASE / "detection" / "ball-detection" / "ball_detection.py")

# Re-export names so the IDE and callers can see them
BallTracker:    type  = _sc.BallTracker
check_ball            = _sc.check_ball
CONF:           float = _sc.CONF
IMGSZ:          int   = _sc.IMGSZ
SMOOTHING_LIVE: float = _sc.SMOOTHING_LIVE
SMOOTHING_HOLD: float = _sc.SMOOTHING_HOLD
MAX_MOVE_LIVE:  int   = _sc.MAX_MOVE_LIVE
MAX_MOVE_HOLD:  int   = _sc.MAX_MOVE_HOLD
cropped_width:  int   = _sc.cropped_width
cropped_height: int   = _sc.cropped_height

kalman_filter         = _bd.kalman_filter


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

                if conf > 0.6:
                    kf.statePre[:2]  = measurement
                    kf.statePost[:2] = measurement

                frame_ball_boxes[str(i)] = [x1, y1, x2, y2, conf]
                have_det = True

        ball_counts.append(1 if have_det else 0)

        if kf_initialized:
            frame_kalman[str(i)] = [px, py]

        if total_frames > 0:
            progress.progress(min((i + 1) / total_frames, 1.0),
                              text=f"Frame {i+1}/{total_frames} — {'ball detected' if have_det else 'no ball'}")

    progress.empty()
    return (clean_frames, frame_ball_boxes, frame_kalman,
            ball_counts, frame_w, frame_h, fps, time.time() - start)


# ─── Smart crop ───────────────────────────────────────────────────────────────

def process_smart_crop_video(video_path, model):
    ball_cls = [k for k, v in model.names.items() if v == "sports ball"][0]

    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None

    h, w      = frame.shape[:2]
    target_w  = min(int(round(h * 9 / 16)), w)
    max_x     = w - target_w
    half_w    = target_w / 2.0
    cx_min, cx_max = half_w, w - half_w
    sx = cropped_width  / target_w
    sy = cropped_height / h

    cropped_frames        = []
    frame_ball_boxes_crop = {}   # {str(i): [x1,y1,x2,y2,conf]} in cropped_width×cropped_height space
    frame_pred_crop       = {}   # {str(i): [px,py]}             in cropped_width×cropped_height space

    crop_cx = None
    tracker = BallTracker()
    fidx    = 0

    progress = st.progress(0, text="Running smart crop...")
    start    = time.time()

    while True:
        if frame is None:
            break

        ball_box = None
        live_cx  = None

        res = model.predict(source=frame, imgsz=IMGSZ, conf=CONF,
                            iou=0.4, verbose=False, classes=[ball_cls])
        r = res[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy  = r.boxes.xyxy.cpu().numpy()   # type: ignore
            confs = r.boxes.conf.cpu().numpy()   # type: ignore
            good  = [i for i in range(len(xyxy))
                     if check_ball(xyxy[i][0], xyxy[i][1], xyxy[i][2], xyxy[i][3], h)]
            if good:
                bi                  = good[int(np.argmax(confs[good]))]
                bx1, by1, bx2, by2  = xyxy[bi]
                dcx, dcy, dcnf      = (bx1+bx2)/2.0, (by1+by2)/2.0, float(confs[bi])
                if not tracker.is_outlier(dcx, dcy, fidx):
                    ball_box = (bx1, by1, bx2, by2, dcnf)
                    tracker.update(fidx, dcx, dcy, dcnf)
                    live_cx = dcx
                else:
                    tracker.tick_miss()
            else:
                tracker.tick_miss()
        else:
            tracker.tick_miss()

        if live_cx is not None:
            sm, mm    = SMOOTHING_LIVE, MAX_MOVE_LIVE
            target_cx = float(np.clip(live_cx, cx_min, cx_max))
        else:
            sm, mm    = SMOOTHING_HOLD, MAX_MOVE_HOLD
            target_cx = crop_cx if crop_cx is not None else w / 2.0

        if crop_cx is None:
            crop_cx = target_cx
        else:
            new_cx = sm * crop_cx + (1 - sm) * target_cx
            delta  = new_cx - crop_cx
            if abs(delta) > mm:
                new_cx = crop_cx + np.sign(delta) * mm
            crop_cx = new_cx

        cx1 = max(0, min(int(round(crop_cx - half_w)), max_x)) if max_x > 0 else 0

        cropped = frame[:, cx1:cx1 + target_w]
        resized = cv2.resize(cropped, (cropped_width, cropped_height))
        cropped_frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

        if ball_box is not None:
            bx1, by1, bx2, by2, bc = ball_box
            frame_ball_boxes_crop[str(fidx)] = [
                float((bx1 - cx1) * sx), float(by1 * sy),
                float((bx2 - cx1) * sx), float(by2 * sy),
                bc,
            ]

        pred = tracker.predict(fidx)
        if pred is not None:
            pcx, pcy, pcnf = pred
            if pcnf == -1.0 or pcnf > 0.05:
                frame_pred_crop[str(fidx)] = [float((pcx - cx1) * sx), float(pcy * sy)]

        if total_frames > 0:
            progress.progress(min((fidx + 1) / total_frames, 1.0),
                              text=f"Frame {fidx+1}/{total_frames}")

        ret, frame = cap.read()
        if not ret:
            break
        fidx += 1

    cap.release()
    progress.empty()
    return (cropped_frames, frame_ball_boxes_crop, frame_pred_crop,
            fps, time.time() - start)