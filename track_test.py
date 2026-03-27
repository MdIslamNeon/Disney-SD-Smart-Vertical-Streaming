# track_test.py
# YOLOv8 detections + ByteTrack tracking (persons only)

import numpy as np
# ---- NumPy 2.x compatibility shim (needed by ByteTrack/yolox) ----
if not hasattr(np, "float"):
    np.float = np.float64
if not hasattr(np, "int"):
    np.int = np.int_
if not hasattr(np, "bool"):
    np.bool = np.bool_
# ------------------------------------------------------------------

import cv2
import time
from ultralytics.models.yolo import YOLO
from types import SimpleNamespace
from yolox.tracker.byte_tracker import BYTETracker

# -------- Config --------
VIDEO_PATH = "ft0_v108_002649_x264.mp4"
MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.25
FPS_ASSUME = 30
# ------------------------

def main():
    model = YOLO(MODEL_PATH)
    args = SimpleNamespace(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        mot20=False,
        frame_rate=FPS_ASSUME
    )
    tracker = BYTETracker(args)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    win = "YOLOv8 + ByteTrack"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    frame_id = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        res = model(frame, verbose=False)[0]
        boxes = res.boxes

        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.empty((0,))
        cls  = boxes.cls.cpu().numpy()  if boxes.cls  is not None else np.empty((0,))

        if len(xyxy) > 0:
            mask = (cls == 0) & (conf >= CONF_THRESH)  # persons only
            xyxy = xyxy[mask]
            conf = conf[mask]
        else:
            xyxy = np.empty((0, 4))
            conf = np.empty((0,))

        if len(xyxy) > 0:
            dets = np.concatenate([xyxy, conf[:, None]], axis=1).astype(np.float32)
        else:
            dets = np.empty((0, 5), dtype=np.float32)

        h, w = frame.shape[:2]
        img_info = (h, w, frame_id)
        img_size = (h, w)
        online_targets = tracker.update(dets, img_info, img_size)

        for t in online_targets:
            x, y, w_t, h_t = t.tlwh
            x1, y1, x2, y2 = int(x), int(y), int(x + w_t), int(y + h_t)
            tid = int(t.track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        fps = frame_id / max(time.time() - t0, 1e-6)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
