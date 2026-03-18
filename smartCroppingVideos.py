import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# -----------------------------
# Project Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
local_videos = BASE_DIR / "local_videos"

output_folder = BASE_DIR / "output_videos_vertical_smart_ball"
output_folder.mkdir(exist_ok=True)

debug_folder = BASE_DIR / "debug_videos_detections"
debug_folder.mkdir(exist_ok=True)

print("Looking in:", local_videos)
print("Exists?", local_videos.exists())

# -----------------------------
# Config (tuned for smoother motion)
# -----------------------------
cropped_width, cropped_height = 540, 960
CONF = 0.15
IMGSZ = 1280

DETECT_EVERY_N = 1   # ✅ run YOLO every frame
SMOOTHING = 0.92     # ✅ stronger smoothing
MAX_MOVE = 25        # ✅ tighter speed cap (pixels/frame). Set None to disable.

# -----------------------------
# YOLO Setup
# -----------------------------
model = YOLO("yolov8m.pt")
ball_class_id = [k for k, v in model.names.items() if v == "sports ball"][0]
print(f"Detected 'sports ball' class ID: {ball_class_id}")

# -----------------------------
# Collect Videos
# -----------------------------
video_files = [
    f for f in local_videos.iterdir()
    if f.is_file() and f.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
]

if not video_files:
    raise FileNotFoundError(f"No video files found in: {local_videos.resolve()}")

print(f"Found {len(video_files)} videos.")

# -----------------------------
# Process Videos
# -----------------------------
for video_path in tqdm(video_files, desc="Smart-cropping videos", unit="video"):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open {video_path.name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        print(f"Could not read first frame from {video_path.name}")
        continue

    h, w = frame.shape[:2]

    # Vertical crop logic
    target_w = int(round(h * 9 / 16))
    target_w = min(target_w, w)
    max_x = w - target_w
    default_x1 = max_x // 2 if max_x > 0 else 0

    out_path = output_folder / f"{video_path.stem}_smartcrop.mp4"
    dbg_path = debug_folder / f"{video_path.stem}_debug.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (cropped_width, cropped_height))
    dbg_writer = cv2.VideoWriter(str(dbg_path), fourcc, fps, (w, h))

    last_x1 = None
    frame_idx = 0

    while True:
        if frame is None:
            break

        # default: hold last crop (or center if never detected)
        x1 = last_x1 if last_x1 is not None else default_x1
        ball_box = None

        # YOLO every frame (DETECT_EVERY_N=1)
        if frame_idx % DETECT_EVERY_N == 0:
            results = model.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF,
                iou=0.4,
                verbose=False,
                classes=[ball_class_id]
            )
            r = results[0]

            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()

                best_i = int(np.argmax(confs))
                x_left, y_top, x_right, y_bottom = xyxy[best_i]
                ball_box = (x_left, y_top, x_right, y_bottom, float(confs[best_i]))

                cx = (x_left + x_right) / 2.0
                raw_x1 = int(round(cx - target_w / 2))

                # clamp raw target
                if max_x > 0:
                    raw_x1 = max(0, min(raw_x1, max_x))
                else:
                    raw_x1 = 0

                # smooth toward raw target
                if last_x1 is None:
                    x1 = raw_x1
                else:
                    x1 = int(SMOOTHING * last_x1 + (1 - SMOOTHING) * raw_x1)

                    # cap movement speed
                    if MAX_MOVE is not None:
                        delta = x1 - last_x1
                        if abs(delta) > MAX_MOVE:
                            x1 = int(last_x1 + np.sign(delta) * MAX_MOVE)

                # update last_x1 only when we had a detection
                last_x1 = x1

        x2 = x1 + target_w

        # crop + resize output
        cropped = frame[:, x1:x2]
        resized = cv2.resize(cropped, (cropped_width, cropped_height))
        writer.write(resized)

        # debug overlay
        dbg = frame.copy()
        cv2.rectangle(dbg, (x1, 0), (x2, h - 1), (0, 255, 255), 3)

        if ball_box is not None:
            x_left, y_top, x_right, y_bottom, c = ball_box
            cv2.rectangle(dbg, (int(x_left), int(y_top)), (int(x_right), int(y_bottom)), (0, 255, 0), 3)
            cv2.putText(
                dbg, f"ball conf={c:.2f}",
                (int(x_left), max(30, int(y_top) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                dbg, "NO BALL DETECTED (holding last position)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        dbg_writer.write(dbg)

        # next frame
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

    cap.release()
    writer.release()
    dbg_writer.release()

    print(f"\nSaved cropped video: {out_path.name}")
    print(f"Saved debug video : {dbg_path.name}")

print("\nAll videos processed.")