# Disney SD — Smart Vertical Basketball Streaming

Automatically reframes landscape basketball video into vertical 9:16 format using YOLOv8 detection and ByteTrack multi-object tracking.

## Project Overview

Most sports live streams are produced in the horizontal 16:9 aspect ratio, but mobile audiences prefer the vertical 9:16 format. Naively cropping a horizontal feed often cuts off the player or the ball, causing viewers to miss critical moments.

This project uses YOLOv8 + ByteTrack to intelligently reframe basketball footage so the action stays centered. A Streamlit web UI lets you upload a video and run any of three analysis modes interactively.

## Features

- **Player Detection** — detects and tracks all players on the court using YOLOv8 + ByteTrack, assigning persistent IDs across frames
- **Ball Detection** — locates the basketball each frame using YOLOv8, with Gaussian filter smoothing to stabilize the ball center trajectory when the ball is briefly out of view
- **Smart Crop** — automatically reframes the video to 9:16 by following the Gaussian-smoothed ball position, keeping crop movement smooth

## Project Structure

```
Disney-SD-Smart-Vertical-Streaming/
├── UI/
│   ├── StreamJup.py              # Main Streamlit app — entry point for the UI
│   ├── detection.py              # Processing engine: process_video, process_ball_video,
│   │                             #   process_smart_crop_video. Imports logic from
│   │                             #   standalone scripts via importlib (single source of truth).
│   ├── html_builders.py          # Builds HTML/JS video player pages (player, ball, smart crop)
│   └── video_utils.py            # render_video() and HTTP server with range-request support
├── cropping/
│   ├── smartCroppingVideos.py    # Source of truth: crop constants + BallTracker class
│   ├── smartCroppingImages.py    # Image-based smart crop (standalone only, not used by UI)
│   ├── reframe_9x16.py           # Alternative player-motion-based reframing (standalone only)
│   └── read-and-crop.py          # Simple centered 9:16 crop with no AI (standalone only)
├── detection/
│   ├── ball-detection/
│   │   └── ball_detection.py     # Source of truth for ball detection helpers + standalone script
│   └── player-detection/
│       └── player_detection.py   # Source of truth for draw_tracked_boxes + standalone script
├── tests/
│   ├── ball_detector_test.py
│   ├── ball_framing_integration_test.py
│   ├── player_tracking_test.py
│   └── read_and_crop_test.py
├── models/
│   ├── yolov8n.pt                # YOLOv8 nano weights
│   ├── yolov8s.pt                # YOLOv8 small weights
│   └── yolov8m.pt                # YOLOv8 medium weights (used by the UI)
├── videos/                       # Input test videos (git-ignored)
├── output/                       # Output videos from standalone scripts
├── sd_disney/                    # Python virtual environment (git-ignored)
├── run_ui.bat                    # One-click launcher for the Streamlit UI
├── requirements.txt
├── tasks.txt                     # Outstanding TODOs and project notes
└── Documentation.txt
```

## Architecture

`UI/detection.py` is the single import point for all detection logic. It loads three external modules at startup via `importlib` so no logic is duplicated:

| Module | Symbols loaded |
|---|---|
| `cropping/smartCroppingVideos.py` | `cropped_width`, `cropped_height` |
| `detection/ball-detection/ball_detection.py` | `GAUSSIAN_SIGMA`, `_choose_best_ball`, `_is_valid_ball_size`, `_reject_spatial_outliers` |
| `detection/player-detection/player_detection.py` | `draw_tracked_boxes` |

### UI Pipeline

1. User uploads an MP4/AVI/MOV via `StreamJup.py`; file path is saved to `st.session_state`
2. User clicks a button → `StreamJup.py` calls the matching pipeline in `detection.py`
3. Results stored in `st.session_state`
4. `video_utils.py` encodes frames to an H.264 MP4 and serves it over HTTP (range-request support for browser seeking)
5. `html_builders.py` builds an HTML5 `<video>` + `<canvas>` overlay with detection data embedded as JSON, synced via `requestAnimationFrame`
6. `StreamJup.py` renders the player in a Streamlit `<iframe>`

**Player Detection** (`process_video`) — YOLOv8 + ByteTrack on every frame (COCO class 0, person). Returns annotated frames, per-frame counts, bounding boxes, and metadata.

**Ball Detection** (`process_ball_video`) — 2-pass: YOLO detects sports ball (COCO class 32) → `_choose_best_ball` picks highest-confidence detection → `_is_valid_ball_size` filters bad boxes → `_reject_spatial_outliers` removes false positives via MAD-sigma on the cx trajectory → gap-fill with `np.interp` → `gaussian_filter1d` smooths cx/cy (sigma=15).

**Smart Crop** (`process_smart_crop_video`) — same ball detection + smoothing pipeline, then derives a crop window x1 from the smoothed ball cx, clips to frame bounds, slices each frame, and resizes to 540×960.

## Setup

**1. Clone the repo**
```bash
git clone <repo-url>
cd Disney-SD-Smart-Vertical-Streaming
```

**2. Create the virtual environment**
```bash
python -m venv sd_disney
```

**3. Activate it**
```bash
# Windows (PowerShell)
.\sd_disney\Scripts\Activate.ps1

# Windows (CMD)
sd_disney\Scripts\activate.bat

# Mac / Linux
source sd_disney/bin/activate
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

## Running the UI

Double-click `run_ui.bat` or run from the project root:
```bash
.\sd_disney\Scripts\python.exe -m streamlit run UI\StreamJup.py
```

Upload a video, then click **Run Player Detection**, **Run Ball Detection**, or **Run Smart Crop**.

## Dependencies

| Package           | Purpose |
|---|---|
| `ultralytics`     | YOLOv8 inference + ByteTrack tracker |
| `opencv-python`   | Video I/O, frame processing, H.264 encoding |
| `streamlit`       | Web UI framework |
| `numpy`           | Array math, interpolation (gap-fill), crop window clamping |
| `scipy`           | Gaussian filtering (ball center smoothing), MAD-based outlier rejection |
| `torch`           | CUDA device detection (standalone scripts) |
| `tqdm`            | Progress bars (standalone scripts) |
| `kagglehub`       | Dataset download (cropping scripts and tests only) |

## Future Work

- **Live stream smart cropping** — `BallTracker` in `smartCroppingVideos.py` is already stateful and frame-by-frame, making it straightforward to adapt for a streaming context.
- **Image-level detection fallback** — `smartCroppingImages.py` provides the per-image detection strategy needed when frames have no ball detection.
- **Alternative crop mode** — `reframe_9x16.py` implements a player-motion-based crop using `SaliencySelector` that could serve as a third mode or as a fallback when ball detection confidence is low.

## Known Issues

- **Pylance "unknown import symbol"** — the `detection/` folder at the project root shadows `UI/detection.py` for the static analyzer. Runtime is unaffected (Streamlit adds `UI/` to `sys.path`). Fix: add a `pyrightconfig.json` at the project root with `{ "extraPaths": ["UI"] }`.

- **ABC logo misdetected as sports ball** — the general-purpose `yolov8m` model occasionally classifies the ABC logo (COCO class 32) as a basketball. Proposed fixes: raise the confidence threshold above 0.20, add a spatial exclusion zone for the logo area, use a sports-specific fine-tuned YOLO model, or use higher-resolution input.