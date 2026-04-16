import os
import tempfile
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ULTRALYTICS_CONFIG_DIR"] = str(os.path.join(tempfile.gettempdir(), "ultralytics"))

import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path: Path = BASE_DIR / "models" / "yolov8m.pt") -> YOLO:
    return YOLO(str(model_path))


def kalman_filter() -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1],
         [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.06
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.3
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf


def run_detection(video_path: Path, model: YOLO, output_folder: Path) -> Path:
    ball_class_id = [k for k, v in model.names.items() if v == "sports ball"][0]
    print(f"Detected 'sports ball' class ID: {ball_class_id}")

    out_path = output_folder / f"{video_path.stem}_ball_detection.mp4"

    results = model.predict(
        source=str(video_path),
        stream=True,
        imgsz=768,
        conf=0.20,
        iou=0.4,
        save=False,
        classes=[ball_class_id],
        device=DEVICE,
    )

    window_name = "YOLO Detections"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)

    kf = kalman_filter()
    frame_i = 0
    t0 = time.time()
    out = None

    for result in results:
        frame = result.orig_img

        if out is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            out = cv2.VideoWriter(str(out_path), fourcc, 30, (w, h))

        pred = kf.predict()
        px, py = int(pred[0]), int(pred[1])
        have_det = False

        if result.boxes is not None and len(result.boxes) > 0:
            confs = result.boxes.conf.cpu().numpy()    # type: ignore[union-attr]
            classes = result.boxes.cls.cpu().numpy()   # type: ignore[union-attr]

            idx = np.argmax(confs)
            if int(classes[idx]) == ball_class_id:
                xyxy = result.boxes.xyxy[idx].cpu().numpy()  # type: ignore[union-attr]
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
                measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
                kf.correct(measurement)
                have_det = True

                if confs[idx] > 0.6:
                    kf.statePre[:2] = measurement
                    kf.statePost[:2] = measurement

        if have_det:
            cv2.circle(frame, (int(cx), int(cy)), 8, (0, 255, 0), -1)

        cv2.circle(frame, (px, py), 8, (0, 0, 255), 2)

        label = "YOLO Detected" if have_det else "Kalman Predict"
        color = (0, 255, 0) if have_det else (0, 0, 255)
        cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

        frame_i += 1
        if frame_i % 30 == 0:
            fps = frame_i / (time.time() - t0)
            print(f"[PERF] {fps:.2f} FPS")

        out.write(frame)

    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    return out_path


def main():
    videos_folder = BASE_DIR / "videos"
    output_folder = BASE_DIR / "output"
    output_folder.mkdir(exist_ok=True)

    video_files = [
        f for f in videos_folder.iterdir()
        if f.is_file() and f.suffix.lower() in {".mp4", ".avi", ".mov"}
    ]
    if not video_files:
        raise FileNotFoundError(f"No video files found in {videos_folder}")

    model = load_model()
    start_time = time.time()

    for video_index, video_path in enumerate(video_files):
        print(f"\nProcessing video {video_index + 1}/{len(video_files)}: {video_path.name}")
        out_path = run_detection(video_path, model, output_folder)
        print(f"Saved: {out_path}")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s ({total_time / 60:.1f} min) for {len(video_files)} file(s).")


if __name__ == "__main__":
    main()