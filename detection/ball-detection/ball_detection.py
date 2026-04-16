import os
import tempfile
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ULTRALYTICS_CONFIG_DIR"] = str(os.path.join(tempfile.gettempdir(), "ultralytics"))

import cv2
import numpy as np
import torch
import time
from scipy.ndimage import gaussian_filter1d
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GAUSSIAN_SIGMA        = 15
BALL_MIN_AREA_FRAC    = 0.0001
BALL_MAX_AREA_FRAC    = 0.02
OUTLIER_MAD_THRESHOLD = 3.0


def load_model(model_path: Path = BASE_DIR / "models" / "yolov8m.pt") -> YOLO:
    return YOLO(str(model_path))


def _choose_best_ball(boxes, ball_class_id: int):
    """Return (box_xyxy, confidence) for the highest-conf sports ball, or None."""
    if boxes is None or len(boxes) == 0:
        return None
    confs        = boxes.conf.cpu().numpy()
    classes      = boxes.cls.cpu().numpy().astype(int)
    ball_indices = np.where(classes == ball_class_id)[0]
    if ball_indices.size == 0:
        return None
    best = ball_indices[np.argmax(confs[ball_indices])]
    return boxes.xyxy[best].cpu().numpy(), float(confs[best])


def _is_valid_ball_size(box_xyxy, frame_w: int, frame_h: int) -> bool:
    """Return True if the bounding box area is within a plausible basketball range."""
    x1, y1, x2, y2 = box_xyxy
    frac = (x2 - x1) * (y2 - y1) / (frame_w * frame_h)
    return BALL_MIN_AREA_FRAC <= frac <= BALL_MAX_AREA_FRAC


def _reject_spatial_outliers(known_indices: list, known_x1s: list,
                              known_cxs: list) -> tuple:
    """Remove detections whose cx is a MAD-sigma outlier across the full clip."""
    if len(known_cxs) < 4:
        return known_indices, known_x1s
    cx_arr    = np.array(known_cxs, dtype=np.float64)
    median_cx = np.median(cx_arr)
    mad       = np.median(np.abs(cx_arr - median_cx))
    if mad < 1e-6:
        return known_indices, known_x1s
    z    = np.abs(cx_arr - median_cx) / (mad / 0.6745)
    mask = z <= OUTLIER_MAD_THRESHOLD
    n_rejected = int((~mask).sum())
    if n_rejected > 0:
        print(f"    Outlier rejection: removed {n_rejected}/{len(known_indices)} detections")
    return (
        [known_indices[i] for i in range(len(known_indices)) if mask[i]],
        [known_x1s[i]     for i in range(len(known_x1s))     if mask[i]],
    )


def run_detection(video_path: Path, model: YOLO, output_folder: Path) -> Path:
    ball_class_id = [k for k, v in model.names.items() if v == "sports ball"][0]
    print(f"Detected 'sports ball' class ID: {ball_class_id}")

    out_path = output_folder / f"{video_path.stem}_ball_detection.mp4"

    # Read video geometry up front
    cap      = cv2.VideoCapture(str(video_path))
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    # ── Pass 1: detect ball in every frame ───────────────────────────────────
    print("  Pass 1: detecting...", end=" ", flush=True)
    t0 = time.time()

    known_indices    = []   # frame indices with a confirmed detection
    known_cxs        = []
    known_cys        = []
    frame_det_boxes  = {}   # {frame_index: [x1, y1, x2, y2, conf]}
    total_frames     = 0

    for i, result in enumerate(model.predict(
            source=str(video_path), classes=[ball_class_id], stream=True,
            save=False, conf=0.20, iou=0.4, imgsz=768, device=DEVICE)):

        detection = _choose_best_ball(result.boxes, ball_class_id)
        if detection is not None:
            box_xyxy, conf = detection
            if _is_valid_ball_size(box_xyxy, frame_w, frame_h):
                x1, y1, x2, y2 = box_xyxy
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                known_indices.append(i)
                known_cxs.append(cx)
                known_cys.append(cy)
                frame_det_boxes[i] = [float(x1), float(y1), float(x2), float(y2), conf]

        total_frames += 1

    print(f"{total_frames} frames collected")

    # ── Pass 1.5: outlier rejection, gap-fill, Gaussian smooth ───────────────
    # Outlier rejection uses cx only (same as integration test)
    known_indices, _ = _reject_spatial_outliers(known_indices, known_cxs, known_cxs)
    # Rebuild cx/cy lists to match surviving indices
    _cx_by_idx = {i: cx for i, cx in zip(known_indices, known_cxs)}
    _cy_by_idx = {i: cy for i, cy in zip(known_indices, known_cys)}
    # Re-filter frame_det_boxes to surviving indices
    frame_det_boxes = {i: frame_det_boxes[i] for i in known_indices if i in frame_det_boxes}
    known_cxs = [_cx_by_idx[i] for i in known_indices]
    known_cys = [_cy_by_idx[i] for i in known_indices]

    all_idx = np.arange(total_frames)
    if len(known_indices) == 0:
        smooth_cxs = np.full(total_frames, frame_w / 2.0)
        smooth_cys = np.full(total_frames, frame_h / 2.0)
    else:
        smooth_cxs = gaussian_filter1d(
            np.interp(all_idx, known_indices, known_cxs),
            sigma=GAUSSIAN_SIGMA, mode="nearest",
        )
        smooth_cys = gaussian_filter1d(
            np.interp(all_idx, known_indices, known_cys),
            sigma=GAUSSIAN_SIGMA, mode="nearest",
        )

    # ── Pass 2: render ────────────────────────────────────────────────────────
    print("  Pass 2: rendering...", end=" ", flush=True)

    cap    = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out    = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, frame_h))

    window_name = "Ball Detection (Gaussian)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)

    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        have_det = frame_i in frame_det_boxes
        px = int(smooth_cxs[frame_i])
        py = int(smooth_cys[frame_i])

        # Draw detection bounding box
        if have_det:
            x1, y1, x2, y2, conf = frame_det_boxes[frame_i]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 8, (0, 255, 0), -1)
            cv2.putText(frame, f"{conf:.0%}", (int(x1), max(0, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw Gaussian-smoothed position
        cv2.circle(frame, (px, py), 12, (0, 0, 255), 2)

        label = "YOLO Detected" if have_det else "Gaussian Smooth"
        color = (0, 255, 0) if have_det else (0, 0, 255)
        cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        out.write(frame)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

        frame_i += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"done | {elapsed:.1f}s | {frame_i / elapsed:.1f} FPS equivalent → {out_path}")
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