import cv2
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
videos_folder = BASE_DIR / "videos"
output_folder = BASE_DIR / "output"
output_folder.mkdir(exist_ok=True)

cropped_width, cropped_height = 540, 960

video_files = [f for f in videos_folder.iterdir() if f.is_file() and f.suffix.lower() == ".mp4"]

if not video_files:
    raise FileNotFoundError(f"No .mp4 files found in {videos_folder}")

print(f"Found {len(video_files)} video(s).")

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

    out_path = output_folder / f"{video_path.stem}_crop_test.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
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
print("All cropped videos are in:", output_folder)