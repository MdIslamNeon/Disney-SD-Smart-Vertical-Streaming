import cv2
import os
from pathlib import Path
import numpy as np
import kagglehub
from tqdm import tqdm
from ultralytics import YOLO

dataset = "gabrielvanzandycke/deepsport-dataset"
cropped_width, cropped_height = 540, 960

dataset_path = Path(kagglehub.dataset_download(dataset))
print("Dataset folder:", dataset_path)

cropped_folder = dataset_path.parent / (dataset_path.name + "_vertical_smart_ball_png")
cropped_folder.mkdir(exist_ok=True)

debug_folder = dataset_path.parent / (dataset_path.name + "_debug_detections")
debug_folder.mkdir(exist_ok=True)

model = YOLO("yolov8m.pt")
ball_class_id = [k for k, v in model.names.items() if v == "sports ball"][0]
print(f"Detected 'sports ball' class ID: {ball_class_id}")

image_files = []
for root, _, files in os.walk(dataset_path):
    for f in files:
        if f.lower().endswith(".png"):
            image_files.append(Path(root) / f)

print(f"Found {len(image_files)} PNG images.")

processed = 0
detected_count = 0
last_x1 = None

CONF = 0.15
IMGSZ = 1280

for img_path in tqdm(image_files, desc="Smart-cropping images", unit="img"):
    frame = cv2.imread(str(img_path))
    if frame is None:
        continue

    h, w = frame.shape[:2]

    target_w = int(round(h * 9 / 16))
    target_w = min(target_w, w)

    max_x = w - target_w
    default_x1 = max_x // 2 if max_x > 0 else 0

    # YOLO
    results = model.predict(
        source=frame,
        imgsz=IMGSZ,
        conf=CONF,
        iou=0.4,
        verbose=False,
        classes=[ball_class_id]
    )
    r = results[0]

    x1 = None
    ball_box = None

    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        best_i = int(np.argmax(confs))
        x_left, y_top, x_right, y_bottom = xyxy[best_i]
        ball_box = (x_left, y_top, x_right, y_bottom, float(confs[best_i]))

        cx = (x_left + x_right) / 2.0
        x1 = int(round(cx - target_w / 2))

        if max_x > 0:
            x1 = max(0, min(x1, max_x))
        else:
            x1 = 0

        last_x1 = x1
        detected_count += 1
    else:
        x1 = last_x1 if last_x1 is not None else default_x1

    x2 = x1 + target_w

    cropped = frame[:, x1:x2]
    resized = cv2.resize(cropped, (cropped_width, cropped_height))

    rel_path = img_path.relative_to(dataset_path)
    out_path = (cropped_folder / rel_path).with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), resized)

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
            dbg, "NO BALL DETECTED (using fallback)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )

    dbg_path = (debug_folder / rel_path).with_suffix(".png")
    dbg_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dbg_path), dbg)

    processed += 1

print(f"\nDone! Processed {processed} images.")
print(f"Ball detected in {detected_count}/{processed} images.")
print("Crops saved in:", cropped_folder)
print("Debug overlays saved in:", debug_folder)
