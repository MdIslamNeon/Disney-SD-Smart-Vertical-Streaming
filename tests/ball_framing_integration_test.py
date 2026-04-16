import os
import tempfile

# For LINUX devices:
# os.environ["QT_QPA_PLATFORM"] = "offscreen"
# os.environ["FORCE_QT"] = "0"
# os.environ["DISPLAY"] = ""
# os.environ["MPLBACKEND"] = "Agg"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ULTRALYTICS_CONFIG_DIR"] = str(os.path.join(tempfile.gettempdir(), "ultralytics"))

import cv2
import numpy as np
import time
import torch
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import kagglehub
from tqdm import tqdm
from ultralytics import YOLO

# ----------------------------- Configuration ----------------------------------
# Set to None to use Kaggle dataset, or a path string to use a single video
INPUT_VIDEO = None
KAGGLE_DATASET = "sarbagyashakya/basketball-51-dataset"
MAX_VIDEOS = 10  # set to None to process all

MODEL_PATH = "models/yolov8m.pt"

INFERENCE_IMAGE_SIZE = 768
CONFIDENCE_THRESHOLD = 0.20
IOU_THRESHOLD = 0.40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_FPS = 30

# 9:16 output resolution
CROP_WIDTH = 540
CROP_HEIGHT = 960

# Gaussian smoothing sigma (in frames).
# At 30fps: sigma=8 (~0.3s) = tight/responsive, sigma=15 (~0.5s) = smooth/cinematic
# Increase for slower panning, decrease for tighter ball-following.
GAUSSIAN_SIGMA = 15

# False positive rejection
# Bounding box area as a fraction of total frame area.
# A basketball typically occupies 0.01%–1% of a 1080p frame.
BALL_MIN_AREA_FRAC = 0.0001   # smaller → likely noise / tiny artefact
BALL_MAX_AREA_FRAC = 0.02     # larger  → likely not a ball (person, logo, etc.)
# Detections whose cx deviates more than this many MAD-sigmas from the
# median cx across the whole clip are treated as spatial outliers.
OUTLIER_MAD_THRESHOLD = 3.0


# ----------------------------- Detection Helpers ------------------------------
def choose_best_ball(boxes, ball_class_id: int):
    """Return (box_xyxy, confidence) for the highest-conf sports ball, or None."""
    if len(boxes) == 0:
        return None
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    ball_indices = np.where(classes == ball_class_id)[0]
    if ball_indices.size == 0:
        return None
    best = ball_indices[np.argmax(confs[ball_indices])]
    return boxes.xyxy[best].cpu().numpy(), float(confs[best])


def is_valid_ball_size(box_xyxy, frame_w: int, frame_h: int) -> bool:
    """Return True if the bounding box area is plausible for a basketball."""
    x1, y1, x2, y2 = box_xyxy
    frac = (x2 - x1) * (y2 - y1) / (frame_w * frame_h)
    return BALL_MIN_AREA_FRAC <= frac <= BALL_MAX_AREA_FRAC


def reject_spatial_outliers(known_indices: list, known_x1s: list,
                             known_cxs: list) -> tuple[list, list]:
    """
    Remove detections whose cx is a statistical outlier across the full clip.

    Uses Median Absolute Deviation (MAD) — robust to the skewed distributions
    that arise when a handful of false positives appear in the corners of the
    frame while the true ball stays near centre.
    """
    if len(known_cxs) < 4:
        return known_indices, known_x1s

    cx_arr = np.array(known_cxs, dtype=np.float64)
    median_cx = np.median(cx_arr)
    mad = np.median(np.abs(cx_arr - median_cx))

    if mad < 1e-6:
        return known_indices, known_x1s

    # Normalise to sigma units (0.6745 = Φ⁻¹(0.75) for a normal distribution)
    z = np.abs(cx_arr - median_cx) / (mad / 0.6745)
    mask = z <= OUTLIER_MAD_THRESHOLD

    n_rejected = int((~mask).sum())
    if n_rejected > 0:
        print(f"    Outlier rejection: removed {n_rejected}/{len(known_indices)} detections")

    return (
        [known_indices[i] for i in range(len(known_indices)) if mask[i]],
        [known_x1s[i]     for i in range(len(known_x1s))     if mask[i]],
    )


# ----------------------------- Pass 1: Detect ---------------------------------
def detect_crop_positions(yolo, ball_class_id: int, input_path: str) -> tuple[list, int, int, int, dict]:
    """
    Run YOLO over the video and return a gap-filled target_x1 per frame,
    plus the video geometry needed for rendering.

    Missed detections are filled by linear interpolation between the nearest
    known positions on either side — Gaussian smoothing in Pass 1.5 then
    curves over those straight segments anyway.

    Returns:
        raw_x1s   : list of float, one per frame — gap-filled but not yet smoothed
        frame_w   : original video width
        frame_h   : original video height
        target_w  : 9:16 crop window width in original pixels
        frame_boxes: dict mapping frame index → box_xyxy for detected frames
    """
    stream = yolo.predict(
        source=input_path,
        stream=True,
        imgsz=INFERENCE_IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        save=False,
        classes=[ball_class_id],
        device=DEVICE,
    )

    frame_w = frame_h = target_w = max_x1 = None
    known_indices = []   # frame indices with a real detection
    known_x1s = []       # corresponding raw crop left-edge positions
    known_cxs = []       # ball centre-x (kept separate for outlier rejection)
    idx_to_box = {}      # frame index → box_xyxy (for bounding box rendering)
    total_frames = 0

    for i, result in enumerate(stream):
        frame = result.orig_img
        h, w = frame.shape[:2]

        if frame_w is None:
            frame_w, frame_h = w, h
            target_w = min(int(h * 9 / 16), w)
            max_x1 = w - target_w

        assert frame_w is not None and frame_h is not None and target_w is not None and max_x1 is not None
        detection = choose_best_ball(result.boxes, ball_class_id)
        if detection is not None:
            box_xyxy, _ = detection
            # Layer 1: size filter — reject implausibly small/large boxes
            if not is_valid_ball_size(box_xyxy, frame_w, frame_h):
                total_frames += 1
                continue
            cx = (box_xyxy[0] + box_xyxy[2]) / 2.0
            x1 = float(np.clip(cx - target_w / 2.0, 0, max_x1))
            known_indices.append(i)
            known_x1s.append(x1)
            known_cxs.append(cx)
            idx_to_box[i] = box_xyxy

        total_frames += 1

    # Layer 2: spatial outlier rejection over the full clip
    known_indices, known_x1s = reject_spatial_outliers(known_indices, known_x1s, known_cxs)

    # Build frame_boxes using only the frames that survived outlier rejection
    frame_boxes = {idx: idx_to_box[idx] for idx in known_indices}

    # Fill every frame index — np.interp clamps to edge values outside known range
    all_indices = np.arange(total_frames)
    if max_x1 is None or total_frames == 0:
        return [], frame_w or 0, frame_h or 0, target_w or 0, {}
    if len(known_indices) == 0:
        # No detections at all — hold center
        raw_x1s = np.full(total_frames, max_x1 / 2.0)
    else:
        raw_x1s = np.interp(all_indices, known_indices, known_x1s)

    assert frame_w is not None and frame_h is not None and target_w is not None
    return raw_x1s.tolist(), frame_w, frame_h, target_w, frame_boxes


# ----------------------------- Pass 1.5: Smooth ------------------------------
def smooth_crop_positions(raw_x1s: list, max_x1: int, sigma: float) -> np.ndarray:
    """
    Apply Gaussian smoothing to the full crop position trajectory.

    Because we have the entire clip, gaussian_filter1d uses both past and
    future frames symmetrically — no causal lag, no phase shift.
    """
    arr = np.array(raw_x1s, dtype=np.float64)
    smoothed = gaussian_filter1d(arr, sigma=sigma, mode="nearest")
    return np.clip(smoothed, 0, max_x1)


# ----------------------------- Pass 2: Render --------------------------------
def render_video(input_path: str, output_path: str, smoothed_x1s: np.ndarray,
                 target_w: int, frame_boxes: dict) -> None:
    """
    Reopen the original video and write each frame using the pre-computed
    smoothed crop positions. Draws the bounding box when a detection exists.
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or OUTPUT_FPS
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (CROP_WIDTH, CROP_HEIGHT))

    for frame_idx, x1 in enumerate(smoothed_x1s):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in frame_boxes:
            x1b, y1b, x2b, y2b = frame_boxes[frame_idx].astype(int)
            cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), (0, 255, 0), 2)
        x1_int = int(round(x1))
        cropped = frame[:, x1_int: x1_int + target_w]
        writer.write(cv2.resize(cropped, (CROP_WIDTH, CROP_HEIGHT)))

    cap.release()
    writer.release()


# ----------------------------- Per-video Orchestrator ------------------------
def process_video(yolo, ball_class_id: int, input_path: str, output_path: str) -> None:
    t0 = time.time()

    # Pass 1 — detection + gap filling, no rendering
    print(f"  Pass 1: detecting...", end=" ", flush=True)
    raw_x1s, frame_w, frame_h, target_w, frame_boxes = detect_crop_positions(
        yolo, ball_class_id, input_path
    )
    max_x1 = frame_w - target_w
    print(f"{len(raw_x1s)} frames collected")

    # Smooth the full trajectory with Gaussian
    smoothed_x1s = smooth_crop_positions(raw_x1s, max_x1, sigma=GAUSSIAN_SIGMA)

    # Pass 2 — render using smoothed positions
    print(f"  Pass 2: rendering...", end=" ", flush=True)
    render_video(input_path, output_path, smoothed_x1s, target_w, frame_boxes)

    elapsed = time.time() - t0
    fps = len(raw_x1s) / elapsed
    print(f"done | {elapsed:.1f}s | {fps:.1f} FPS equivalent → {output_path}")


# ----------------------------- Main ------------------------------------------
def main() -> None:
    yolo = YOLO(MODEL_PATH)
    ball_class_id = next(k for k, v in yolo.names.items() if v == "sports ball")
    print(f"Ball class ID: {ball_class_id}")

    # Single video mode
    if INPUT_VIDEO is not None:
        process_video(yolo, ball_class_id, INPUT_VIDEO, "integration_output.mp4")
        return

    # Batch mode — Kaggle dataset
    dataset_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"Dataset: {dataset_path}")

    output_folder = dataset_path.parent / (dataset_path.name + "_vertical_tracked")
    output_folder.mkdir(exist_ok=True)

    video_files = sorted([
        Path(root) / f
        for root, _, files in os.walk(dataset_path)
        for f in files if f.lower().endswith(".mp4")
    ])

    if MAX_VIDEOS is not None:
        video_files = video_files[:MAX_VIDEOS]

    print(f"Processing {len(video_files)} videos → {output_folder}\n")

    for video_path in tqdm(video_files, desc="Videos", unit="video"):
        rel = video_path.relative_to(dataset_path)
        out_path = (output_folder / rel).with_suffix(".mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        process_video(yolo, ball_class_id, str(video_path), str(out_path))

    print(f"\nAll done. Output folder: {output_folder}")


if __name__ == "__main__":
    main()
