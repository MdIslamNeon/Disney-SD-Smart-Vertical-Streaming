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
#! TODO: BYTETracker is imported from the external ByteTrack repo (yolox package),
# which requires: yolox @ git+https://github.com/ifzhang/ByteTrack.git in requirements.txt
# This could be migrated to Ultralytics' built-in ByteTrack via model.track(tracker="bytetrack.yaml"),
# which would remove the external dependency entirely (see tests/player_tracking_test.py for reference).
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
    # Ensure min size > 0
    if x2 <= x1: x2 = min(W - 1, x1 + 1)
    if y2 <= y1: y2 = min(H - 1, y1 + 1)
    return int(x1), int(y1), int(x2), int(y2)

def aspect_fit_center(cx, cy, tgt_aspect, width_hint, W, H):
    """
    Build a rectangle centered on (cx,cy) with target aspect (w/h) and width ~ width_hint,
    then clamp to image bounds.
    """
    w = clamp(width_hint, 64, W)  # sanity
    h = int(round(w / tgt_aspect))
    if h > H:
        h = H
        w = int(round(h * tgt_aspect))

    x1 = int(round(cx - w / 2))
    y1 = int(round(cy - h / 2))
    x2 = x1 + w
    y2 = y1 + h

    # clamp
    x1, y1, x2, y2 = clamp_rect(x1, y1, x2, y2, W, H)

    # recentre after clamping if we hit edges (optional: helps keep subject centered near borders)
    w = x2 - x1
    h = y2 - y1
    cx = clamp(cx, w // 2, W - w // 2)
    cy = clamp(cy, h // 2, H - h // 2)
    x1 = int(cx - w // 2); y1 = int(cy - h // 2)
    x2 = x1 + w; y2 = y1 + h
    x1, y1, x2, y2 = clamp_rect(x1, y1, x2, y2, W, H)
    return x1, y1, x2, y2

# ------------------ selection logic ------------------

class SaliencySelector:
    """
    Maintains simple per-track motion statistics to pick 'involved' players:
    we score each track by average speed over a short window; pick top-K.
    """
    def __init__(self, win=12):  # ~0.4s at 30fps
        self.prev_center = {}               # tid -> (x,y)
        self.speeds = defaultdict(lambda: deque(maxlen=win))  # tid -> last-N speeds (px/frame)

    def update_and_rank(self, tracks):
        for t in tracks:
            tid = int(t.track_id)
            x, y, w, h = t.tlwh
            cx, cy = x + w / 2, y + h / 2
            if tid in self.prev_center:
                px, py = self.prev_center[tid]
                v = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                self.speeds[tid].append(v)
            self.prev_center[tid] = (cx, cy)

        # compute average speed score
        ranked = []
        for t in tracks:
            tid = int(t.track_id)
            s = np.mean(self.speeds[tid]) if len(self.speeds[tid]) else 0.0
            ranked.append((s, t))
        ranked.sort(key=lambda z: z[0], reverse=True)
        return ranked

# ------------------ main processing ------------------

def process(args):
    # NumPy 2.x compatibility (ByteTrack sometimes expects old aliases)
    if not hasattr(np, "float"):
        np.float = np.float64  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = np.int_  # type: ignore[attr-defined]
    if not hasattr(np, "bool"):
        np.bool = np.bool_  # type: ignore[attr-defined]

    # Load detector
    model = YOLO(args.weights)

    # Init ByteTrack
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

    # Vertical output (default 1080x1920)
    out_H = args.out_h
    out_W = int(out_H * 9 / 16)
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if args.codec == "mp4v" else "avc1")) # type: ignore
    writer = cv2.VideoWriter(args.output, fourcc, out_fps, (out_W, out_H))

    # virtual camera state
    tgt_aspect = out_W / out_H  # 9/16
    center_smooth = None
    width_smooth = None
    selector = SaliencySelector(win=12)

    frame_id = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # detection
        res = model(frame, verbose=False)[0]
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.empty((0,))
        cls  = boxes.cls.cpu().numpy()  if boxes.cls  is not None else np.empty((0,))

        # persons only
        if len(xyxy) > 0:
            mask = (cls == 0) & (conf >= args.conf)
            xyxy = xyxy[mask]
            conf = conf[mask]
        else:
            xyxy = np.empty((0, 4))
            conf = np.empty((0,))

        # dets -> ByteTrack format
        if len(xyxy) > 0:
            dets = np.concatenate([xyxy, conf[:, None]], axis=1).astype(np.float32)
        else:
            dets = np.empty((0, 5), dtype=np.float32)

        img_info = (frame.shape[0], frame.shape[1], frame_id)
        img_size = (frame.shape[0], frame.shape[1])
        tracks = tracker.update(dets, img_info, img_size)

        # ---- pick involved players (top-K by recent speed) ----
        ranked = selector.update_and_rank(tracks)
        top = [t for _, t in ranked[:args.topk]]
        # fallback: if none, keep previous camera
        if len(top) == 0:
            # use a gentle decay to last width
            width_smooth = ema(width_smooth, width_smooth if width_smooth else src_W * 0.6, 0.9)
            cx, cy = (src_W / 2, src_H / 2)
        else:
            # weighted center (by bbox area so closer/larger players matter a bit more)
            centers = []
            weights = []
            xs, ys = [], []
            for t in top:
                x, y, w, h = t.tlwh
                cx_i, cy_i = x + w / 2, y + h / 2
                centers.append((cx_i, cy_i))
                area = max(1.0, w * h)
                weights.append(area)
                xs.extend([x, x + w])
                ys.extend([y, y + h])

            weights = np.array(weights, dtype=np.float64)
            weights /= weights.sum()
            cx = float(np.sum([c[0] * w for c, w in zip(centers, weights)]))
            cy = float(np.sum([c[1] * w for c, w in zip(centers, weights)]))

            # desired width: proportional to spread of top tracks (keep some padding)
            if len(xs) >= 2:
                spread_x = max(xs) - min(xs)
                spread_y = max(ys) - min(ys)
                # pad to keep context; keep aspect later
                pad = 1.6  # >1 widens view a bit
                width_hint = max(spread_x, spread_y * (out_W / out_H)) * pad
            else:
                width_hint = src_W * 0.5

            # clamp zoom range
            min_width = src_W * 0.35  # max zoom-in
            max_width = src_W * 0.85  # min zoom-in (wider view)
            width_hint = clamp(width_hint, min_width, max_width)

            # smoothing + deadband
            center_deadband = 12  # px; ignore tiny movement
            if center_smooth is None:
                center_smooth = (cx, cy)
            else:
                px, py = center_smooth
                if abs(cx - px) < center_deadband: cx = px
                if abs(cy - py) < center_deadband: cy = py

            # EMA smoothing (lower alpha = smoother)
            center_smooth = (ema(center_smooth[0], cx, args.center_alpha),
                             ema(center_smooth[1], cy, args.center_alpha))
            width_smooth = ema(width_smooth, width_hint, args.zoom_alpha)

        # build crop with smoothed params
        if center_smooth is None:
            center_smooth = (src_W / 2, src_H / 2)
        if width_smooth is None:
            width_smooth = src_W * 0.6

        x1, y1, x2, y2 = aspect_fit_center(center_smooth[0], center_smooth[1],
                                           tgt_aspect, width_smooth, src_W, src_H)
        crop = frame[y1:y2, x1:x2]
        vout = cv2.resize(crop, (out_W, out_H), interpolation=cv2.INTER_LINEAR)

        # optional debug overlays (IDs and crop)
        if args.debug:
            for t in tracks:
                x, y, w, h = t.tlwh
                p1 = (int(x), int(y)); p2 = (int(x + w), int(y + h))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                cv2.putText(frame, f"ID {int(t.track_id)}", (p1[0], max(0, p1[1]-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            # draw crop rect on a tiny thumbnail
            thumb = cv2.resize(frame, (640, int(640 * src_H / src_W)))
            sx = 640 / src_W; sy = (640 * src_H / src_W) / src_H
            rx1, ry1, rx2, ry2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
            cv2.rectangle(thumb, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
            # stack side-by-side for preview
            disp = np.zeros((out_H, out_W + 640, 3), dtype=np.uint8)
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
    ap.add_argument("--input", "-i", default="ft0_v108_002649_x264.mp4", help="input 16:9 video")
    ap.add_argument("--output", "-o", default="out_vertical.mp4", help="output 9:16 video")
    ap.add_argument("--weights", default="models/yolov8n.pt", help="YOLOv8 weights")
    ap.add_argument("--fps", type=int, default=0, help="override output fps (0 = inherit)")
    ap.add_argument("--out_h", type=int, default=1920, help="vertical output height (default 1920)")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO person confidence threshold")
    ap.add_argument("--topk", type=int, default=3, help="how many fastest tracks to follow")
    ap.add_argument("--center_alpha", type=float, default=0.25, help="EMA alpha for center (lower = smoother)")
    ap.add_argument("--zoom_alpha", type=float, default=0.15, help="EMA alpha for zoom (lower = smoother)")
    ap.add_argument("--codec", choices=["mp4v","avc1"], default="mp4v", help="VideoWriter fourcc")
    ap.add_argument("--debug", action="store_true", help="show preview with overlays")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process(args)
