# reframe_9x16.py
# Auto-zoom/reframe a horizontal basketball video to 9:16 around the "involved" players.
# Involved players proxy = fastest-moving tracks over a short window.

import argparse
import time
from types import SimpleNamespace
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker

# ------------------ small helpers ------------------

def ema(prev, new, alpha):
    return new if prev is None else (alpha * new + (1 - alpha) * prev)

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def clamp_rect(x1, y1, x2, y2, W, H):
    x1 = clamp(x1, 0, W - 1)
    y1 = clamp(y1, 0, H - 1)
    x2 = clamp(x2, 0, W - 1)
    y2 = clamp(y2, 0, H - 1)
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)
    return int(x1), int(y1), int(x2), int(y2)

def aspect_fit_center(cx, cy, tgt_aspect, width_hint, W, H):
    w = clamp(width_hint, 64, W)
    h = int(round(w / tgt_aspect))
    if h > H:
        h = H
        w = int(round(h * tgt_aspect))

    x1 = int(round(cx - w / 2))
    y1 = int(round(cy - h / 2))
    x2 = x1 + w
    y2 = y1 + h

    x1, y1, x2, y2 = clamp_rect(x1, y1, x2, y2, W, H)

    w = x2 - x1
    h = y2 - y1
    cx = clamp(cx, w // 2, W - w // 2)
    cy = clamp(cy, h // 2, H - h // 2)

    x1 = int(cx - w // 2)
    y1 = int(cy - h // 2)
    x2 = x1 + w
    y2 = y1 + h
    return clamp_rect(x1, y1, x2, y2, W, H)

# ------------------ selection logic ------------------

class SaliencySelector:
    def __init__(self, win=12):
        self.prev_center = {}
        self.speeds = defaultdict(lambda: deque(maxlen=win))
        self.last_seen = defaultdict(int)

    def update_and_rank(self, tracks):
        for t in tracks:
            tid = int(t.track_id)
            x, y, w, h = t.tlwh
            cx, cy = x + w/2, y + h/2
            self.last_seen[tid] += 1

            if tid in self.prev_center:
                px, py = self.prev_center[tid]
                v = ((cx - px)**2 + (cy - py)**2)**0.5
                self.speeds[tid].append(v)

            self.prev_center[tid] = (cx, cy)

        ranked = []
        for t in tracks:
            tid = int(t.track_id)
            s = np.mean(self.speeds[tid]) if len(self.speeds[tid]) else 0.0
            stable = min(1.0, self.last_seen[tid] / 30)
            ranked.append((s * (0.6 + 0.4 * stable), t))
        ranked.sort(key=lambda z: z[0], reverse=True)
        return ranked

# ------------------ main processing ------------------

def process(args):

    # NumPy alias patches for ByteTrack
    if not hasattr(np, "float"): np.float = np.float64
    if not hasattr(np, "int"):   np.int   = np.int_
    if not hasattr(np, "bool"):  np.bool  = np.bool_

    model = YOLO(args.weights)

    bargs = SimpleNamespace(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        mot20=False,
        frame_rate=args.fps if args.fps > 0 else 30
    )
    tracker = BYTETracker(bargs)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    src_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or (args.fps if args.fps > 0 else 30)
    out_fps = args.fps if args.fps > 0 else src_fps

    out_H = args.out_h
    out_W = int(out_H * 9/16)

    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if args.codec=="mp4v" else "avc1"))
    writer = cv2.VideoWriter(args.output, fourcc, out_fps, (out_W, out_H))

    tgt_aspect = out_W / out_H
    center_smooth = None
    width_smooth = None
    selector = SaliencySelector(win=12)

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        res = model(frame, verbose=False)[0]
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0,4))
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.empty((0,))
        cls  = boxes.cls.cpu().numpy()  if boxes.cls  is not None else np.empty((0,))

        if len(xyxy) > 0:
            mask = (cls == 0) & (conf >= args.conf)
            xyxy = xyxy[mask]
            conf = conf[mask]
        else:
            xyxy = np.empty((0,4))
            conf = np.empty((0,))

        if len(xyxy) > 0:
            dets = np.concatenate([xyxy, conf[:,None]], axis=1).astype(np.float32)
        else:
            dets = np.empty((0,5), dtype=np.float32)

        img_info = (frame.shape[0], frame.shape[1], frame_id)
        img_size = (frame.shape[0], frame.shape[1])
        tracks = tracker.update(dets, img_info, img_size)

        ranked = selector.update_and_rank(tracks)
        top = [t for _, t in ranked[:args.topk]]

        if len(top) == 0:
            width_smooth = ema(width_smooth, width_smooth if width_smooth else src_W*0.6, 0.92)
            cx, cy = (src_W/2, src_H/2)

        else:
            centers = []
            xs, ys = [], []
            for t in top:
                x, y, w, h = t.tlwh
                cx_i, cy_i = x + w/2, y + h/2
                centers.append((cx_i, cy_i))
                xs.extend([x, x+w])
                ys.extend([y, y+h])

            # ---- minimal change: use geometric median approximation ----
            arr = np.array(centers)
            cx = float(np.median(arr[:,0]))
            cy = float(np.median(arr[:,1]))

            spread_x = max(xs) - min(xs)
            spread_y = max(ys) - min(ys)

            # slightly reduced padding (less over-wide)
            pad = 1.35
            
            width_hint = max(spread_x, spread_y * (out_W/out_H)) * pad

            # --- small improvement: ensure minimum subject margin ---
            width_hint += src_W * 0.05
            
            min_width = src_W * 0.35
            max_width = src_W * 0.85
            width_hint = clamp(width_hint, min_width, max_width)

            # update 1: change deadband from 12 to 20 to ignore small movements
            center_deadband = 20
            if center_smooth is None:
                center_smooth = (cx, cy)
            else:
                px, py = center_smooth
                if abs(cx - px) < center_deadband: cx = px
                if abs(cy - py) < center_deadband: cy = py

            alpha_x = args.center_alpha
            alpha_y = args.center_alpha * 0.5  # slower vertical

            center_smooth = (
                ema(center_smooth[0], cx, alpha_x),
                ema(center_smooth[1], cy, alpha_y)
            )


            # slightly smoother zoom
            width_smooth = ema(width_smooth, width_hint, args.zoom_alpha * 0.85)

        if center_smooth is None:
            center_smooth = (src_W/2, src_H/2)
        if width_smooth is None:
            width_smooth = src_W * 0.6

        x1, y1, x2, y2 = aspect_fit_center(center_smooth[0], center_smooth[1],
                                           tgt_aspect, width_smooth, src_W, src_H)

        crop = frame[y1:y2, x1:x2]

        # ---- small improvement: higher quality upscale ----
        vout = cv2.resize(crop, (out_W, out_H), interpolation=cv2.INTER_CUBIC)

        if args.debug:
            for t in tracks:
                x, y, w, h = t.tlwh
                cv2.rectangle(frame, (int(x),int(y)), (int(x+w),int(y+h)), (0,255,0), 2)
                cv2.putText(frame, f"ID {int(t.track_id)}",
                            (int(x), max(0, int(y)-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            thumb = cv2.resize(frame, (640, int(640*src_H/src_W)))
            sx = 640/src_W
            sy = (640*src_H/src_W)/src_H
            rx1, ry1, rx2, ry2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
            cv2.rectangle(thumb, (rx1,ry1), (rx2,ry2), (0,0,255), 2)

            disp = np.zeros((out_H, out_W+640, 3), dtype=np.uint8)
            disp[:, :out_W] = vout
            thH = thumb.shape[0]
            disp[:thH, out_W:out_W+640] = thumb
            cv2.imshow("vertical | debug", disp)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

        writer.write(vout)

    cap.release()
    writer.release()
    if args.debug:
        cv2.destroyAllWindows()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default="ft0_v108_002649_x264.mp4")
    ap.add_argument("--output", "-o", default="out_vertical_1.mp4")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--fps", type=int, default=0)
    ap.add_argument("--out_h", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--topk", type=int, default=3)
    # update 1: decrease default from .25 to .1 to change speed of camera
    ap.add_argument("--center_alpha", type=float, default=0.1)
    ap.add_argument("--zoom_alpha", type=float, default=0.07)
    ap.add_argument("--codec", choices=["mp4v","avc1"], default="mp4v")
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process(args)
