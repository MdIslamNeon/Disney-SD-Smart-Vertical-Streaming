import cv2
import os
from pathlib import Path
import kagglehub
from tqdm import tqdm   # ✅ add this

dataset = "sarbagyashakya/basketball-51-dataset"
cropped_width, cropped_height = 540, 960

dataset_path = Path(kagglehub.dataset_download(dataset))
print("Dataset folder:", dataset_path)

cropped_folder = dataset_path.parent / (dataset_path.name + "_vertical_and_moving")
cropped_folder.mkdir(exist_ok=True)

video_files = []
for root, _, files in os.walk(dataset_path):
    for f in files:
        if f.lower().endswith(".mp4"):
            video_files.append(Path(root) / f)

print(f"Found {len(video_files)} videos.")

# set to only 10 to not run the 10K+ videos
video_files = video_files[:10]

processed = 0

for video_path in tqdm(video_files, desc="Cropping videos", unit="video"):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        continue

    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if not ret:
        cap.release()
        continue

    h, w = frame.shape[:2]

    target_w = int(round(h * 9 / 16))
    target_w = min(target_w, w)

    max_x = w - target_w
    x1 = max_x // 2
    direction = 1
    move_ppf = 10

    rel_path = video_path.relative_to(dataset_path)
    out_path = (cropped_folder / rel_path).with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (cropped_width, cropped_height))

    while True:
        if max_x > 0:
            x1 += direction * move_ppf
            if x1 <= 0:
                x1 = 0
                direction = 1
            elif x1 >= max_x:
                x1 = max_x
                direction = -1

        x2 = x1 + target_w

        cropped = frame[:, x1:x2]
        resized = cv2.resize(cropped, (cropped_width, cropped_height))
        writer.write(resized)

        ret, frame = cap.read()
        if not ret:
            break

    cap.release()
    writer.release()
    processed += 1

print(f"\nDone! Processed {processed}/{len(video_files)} videos.")
print("All cropped videos are in:", cropped_folder)