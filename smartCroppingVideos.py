import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from collections import deque

BASE_DIR = Path(__file__).resolve().parent
local_videos = BASE_DIR / "local_videos"

output_folder = BASE_DIR / "output_videos_vertical_smart_ball"
output_folder.mkdir(exist_ok=True)

debug_folder = BASE_DIR / "debug_videos_detections"
debug_folder.mkdir(exist_ok=True)

cropped_width, cropped_height = 540, 960

CONF = 0.15
IMGSZ = 1280
DETECT_EVERY_N = 1

BALL_MIN_PX = 10
BALL_MAX_PX = 220
BALL_MAX_ASPECT = 1.6

SMOOTHING_LIVE = 0.60
SMOOTHING_HOLD = 0.98
MAX_MOVE_LIVE = 120
MAX_MOVE_HOLD = 10

HISTORY_LEN = 20
MIN_HISTORY = 5
PRED_DECAY = 0.90
MAX_PRED_FRAMES = 25
MAX_QUAD_COEFF = 0.08

OUTLIER_BASE = 180
OUTLIER_WIDEN = 20
OUTLIER_MAX = 500

WEIGHT_POW = 2.0


class BallTracker:
    def __init__(self):
        self.detections = deque(maxlen=HISTORY_LEN)
        self.missed = 0
        self.last_conf = 0.0
        self._px = self._py = None
        self._fit_frame = -1

    def update(self, frame_idx, cx, cy, conf):
        self.detections.append((frame_idx, cx, cy))
        self.missed = 0
        self.last_conf = conf
        self._fit_frame = -1

    def tick_miss(self):
        self.missed += 1

    def predict(self, frame_idx):
        if len(self.detections) < MIN_HISTORY:
            return None
        if self.missed > MAX_PRED_FRAMES:
            d = self.detections[-1]
            return float(d[1]), float(d[2]), -1.0
        px, py = self._fit(frame_idx)
        conf = PRED_DECAY ** self.missed
        return float(np.polyval(px, frame_idx)), float(np.polyval(py, frame_idx)), conf

    def is_outlier(self, cx, cy, frame_idx):
        pred = self.predict(frame_idx)
        if pred is None:
            return False
        pcx, pcy, pconf = pred
        if pconf == -1.0 or pconf < 0.2:
            return False
        thresh = min(OUTLIER_BASE + self.missed * OUTLIER_WIDEN, OUTLIER_MAX)
        return float(np.hypot(cx - pcx, cy - pcy)) > thresh

    def arc_points(self, frame_idx, n=15):
        if len(self.detections) < MIN_HISTORY:
            return []
        px, py = self._fit(frame_idx)
        return [(float(np.polyval(px, f)), float(np.polyval(py, f)))
                for f in range(frame_idx, frame_idx + n)]

    @property
    def deg(self):
        return 2 if (self._px is not None and len(self._px) == 3) else 1

    def _fit(self, frame_idx):
        if self._fit_frame == frame_idx and self._px is not None:
            return self._px, self._py
        n = len(self.detections)
        fs = np.array([d[0] for d in self.detections], dtype=float)
        xs = np.array([d[1] for d in self.detections], dtype=float)
        ys = np.array([d[2] for d in self.detections], dtype=float)
        w = np.arange(1, n + 1, dtype=float) ** WEIGHT_POW
        deg = 2 if (n >= 5 and self.missed <= MAX_PRED_FRAMES) else 1
        try:
            px = np.polyfit(fs, xs, deg, w=w)
            py = np.polyfit(fs, ys, deg, w=w)
        except np.linalg.LinAlgError:
            px = np.polyfit(fs, xs, 1, w=w)
            py = np.polyfit(fs, ys, 1, w=w)
        if deg == 2 and (abs(px[0]) > MAX_QUAD_COEFF or abs(py[0]) > MAX_QUAD_COEFF):
            px = np.polyfit(fs, xs, 1, w=w)
            py = np.polyfit(fs, ys, 1, w=w)
        self._px, self._py, self._fit_frame = px, py, frame_idx
        return px, py


def check_ball(x1, y1, x2, y2, fh):
    bw, bh = x2 - x1, y2 - y1
    if bw < BALL_MIN_PX or bh < BALL_MIN_PX:
        return False
    if bw > BALL_MAX_PX or bh > BALL_MAX_PX:
        return False
    if max(bw, bh) / max(min(bw, bh), 1) > BALL_MAX_ASPECT:
        return False
    if (y1 + y2) / 2.0 < fh * 0.08:
        return False
    return True


model = YOLO("yolov8m.pt")
ball_cls = [k for k, v in model.names.items() if v == "sports ball"][0]

video_files = [f for f in local_videos.iterdir() if f.is_file() and f.suffix.lower() == ".mp4"]
if not video_files:
    raise FileNotFoundError(f"no videos in {local_videos}")

for video_path in tqdm(video_files, desc="processing", unit="video"):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        continue

    h, w = frame.shape[:2]
    target_w = min(int(round(h * 9 / 16)), w)
    max_x = w - target_w
    half_w = target_w / 2.0
    cx_min, cx_max = half_w, w - half_w

    out_path = output_folder / f"{video_path.stem}_smartcrop_v5.mp4"
    dbg_path = debug_folder / f"{video_path.stem}_debug_v5.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (cropped_width, cropped_height))
    dbg_writer = cv2.VideoWriter(str(dbg_path), fourcc, fps, (w, h))

    crop_cx = None
    fidx = 0
    tracker = BallTracker()

    while True:
        if frame is None:
            break

        ball_box = None
        rejected = []
        status = "holding"
        live_cx = None

        if fidx % DETECT_EVERY_N == 0:
            res = model.predict(source=frame, imgsz=IMGSZ, conf=CONF,
                                iou=0.4, verbose=False, classes=[ball_cls])
            r = res[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()

                good = [i for i in range(len(xyxy)) if check_ball(*xyxy[i], h)]
                bad = [i for i in range(len(xyxy)) if i not in good]
                for i in bad:
                    rejected.append(tuple(xyxy[i]))

                if good:
                    bi = good[int(np.argmax(confs[good]))]
                    x1, y1, x2, y2 = xyxy[bi]
                    dcx = (x1 + x2) / 2.0
                    dcy = (y1 + y2) / 2.0
                    dcnf = float(confs[bi])

                    if tracker.is_outlier(dcx, dcy, fidx):
                        rejected.append((x1, y1, x2, y2))
                        tracker.tick_miss()
                        status = "outlier"
                    else:
                        ball_box = (x1, y1, x2, y2, dcnf)
                        tracker.update(fidx, dcx, dcy, dcnf)
                        live_cx = dcx
                        status = f"ball {dcnf:.2f}"
                else:
                    tracker.tick_miss()
            else:
                tracker.tick_miss()

        if live_cx is not None:
            sm = SMOOTHING_LIVE
            mm = MAX_MOVE_LIVE
            target_cx = float(np.clip(live_cx, cx_min, cx_max))
        else:
            sm = SMOOTHING_HOLD
            mm = MAX_MOVE_HOLD
            target_cx = crop_cx if crop_cx is not None else w / 2.0

        if crop_cx is None:
            crop_cx = target_cx
        else:
            new_cx = sm * crop_cx + (1 - sm) * target_cx
            delta = new_cx - crop_cx
            if abs(delta) > mm:
                new_cx = crop_cx + np.sign(delta) * mm
            crop_cx = new_cx

        x1 = max(0, min(int(round(crop_cx - half_w)), max_x)) if max_x > 0 else 0
        x2 = x1 + target_w

        writer.write(cv2.resize(frame[:, x1:x2], (cropped_width, cropped_height)))

        dbg = frame.copy()
        cv2.rectangle(dbg, (x1, 0), (x2, h - 1), (0, 255, 255), 3)

        for rb in rejected:
            cv2.rectangle(dbg, (int(rb[0]), int(rb[1])), (int(rb[2]), int(rb[3])), (0, 80, 255), 2)
            cv2.putText(dbg, "x", (int(rb[0]), max(20, int(rb[1]) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)

        if ball_box is not None:
            bx1, by1, bx2, by2, bc = ball_box
            cv2.rectangle(dbg, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 255, 0), 4)
            cv2.putText(dbg, f"{bc:.2f}", (int(bx1), max(40, int(by1) - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

        pred = tracker.predict(fidx)
        if pred is not None:
            pcx, pcy, pcnf = pred
            if pcnf == -1.0 or pcnf > 0.05:
                col = (255, 100, 0) if pcnf != -1.0 else (150, 100, 100)
                cv2.circle(dbg, (int(pcx), int(pcy)), 18, col, 3)

        arc, prev_pt = tracker.arc_points(fidx, 15), None
        for tx, ty in arc:
            pt = (int(np.clip(tx, 0, w - 1)), int(np.clip(ty, 0, h - 1)))
            if prev_pt:
                cv2.line(dbg, prev_pt, pt, (255, 80, 0), 2)
            cv2.circle(dbg, pt, 4, (255, 80, 0), -1)
            prev_pt = pt

        scol = (0, 255, 0) if "ball" in status else (180, 180, 0)
        cv2.putText(dbg, status, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, scol, 3)
        cv2.putText(dbg, f"f={fidx}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

        dbg_writer.write(dbg)

        ret, frame = cap.read()
        if not ret:
            break
        fidx += 1

    cap.release()
    writer.release()
    dbg_writer.release()
    print(f"done: {out_path.name}")