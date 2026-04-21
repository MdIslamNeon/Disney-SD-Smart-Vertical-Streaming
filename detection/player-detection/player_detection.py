import os
import tempfile
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ULTRALYTICS_CONFIG_DIR"] = str(os.path.join(tempfile.gettempdir(), "ultralytics"))

import cv2
import time
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def draw_tracked_boxes(frame, boxes, track_ids):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick, pad = 0.6, 2, 4
    for box, tid in zip(boxes, track_ids):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID {int(tid)}"
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        ly2 = max(th + pad * 2, y1)
        ly1 = ly2 - (th + pad * 2)
        cv2.rectangle(frame, (x1, ly1), (x1 + tw + pad * 2, ly2), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + pad, ly2 - pad),
                    font, scale, (0, 0, 0), thick)
    return frame


def load_model(model_path: Path = BASE_DIR / "models" / "yolov8m.pt") -> YOLO:
    return YOLO(str(model_path))


def get_video_files(videos_folder: Path) -> list[Path]:
    return [
        f for f in videos_folder.iterdir()
        if f.is_file() and f.suffix.lower() in {".mp4", ".avi", ".mov"}
    ]


def run_tracking(video_path: Path, model: YOLO, output_folder: Path) -> Path:
    out_path = output_folder / f"{video_path.stem}_player_tracking.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path.name}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, frame_h))

    win = "ByteTrack Player Detection"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    results = model.track(
        source=str(video_path),
        classes=[0],
        stream=True,
        imgsz=416,
        conf=0.35,
        iou=0.8,
        persist=True,
        tracker="bytetrack.yaml",
        device=DEVICE,
        verbose=False,
    )

    frame_id = 0
    t0 = time.time()

    for result in results:
        frame = result.orig_img.copy()

        if result.boxes is not None and result.boxes.id is not None:
            draw_tracked_boxes(
                frame,
                result.boxes.xyxy.cpu().numpy(),  # type: ignore[union-attr]
                result.boxes.id.cpu().numpy(),    # type: ignore[union-attr]
            )

        frame_id += 1
        live_fps = frame_id / max(time.time() - t0, 1e-6)
        cv2.putText(frame, f"FPS: {live_fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if frame_id % 30 == 0:
            print(f"  Frame {frame_id} — {live_fps:.1f} FPS")

        writer.write(frame)
        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    writer.release()
    cv2.destroyAllWindows()
    return out_path


def main():
    videos_folder = BASE_DIR / "videos"
    output_folder = BASE_DIR / "output"
    output_folder.mkdir(exist_ok=True)

    video_files = get_video_files(videos_folder)
    if not video_files:
        raise FileNotFoundError(f"No video files found in {videos_folder}")

    print(f"Found {len(video_files)} video(s).")

    model = load_model()
    start_time = time.time()

    for video_index, video_path in enumerate(video_files):
        print(f"\nProcessing video {video_index + 1}/{len(video_files)}: {video_path.name}")
        try:
            out_path = run_tracking(video_path, model, output_folder)
            print(f"Saved: {out_path}")
        except RuntimeError as e:
            print(f"Skipping: {e}")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s ({total_time / 60:.1f} min) for {len(video_files)} file(s).")


if __name__ == "__main__":
    main()