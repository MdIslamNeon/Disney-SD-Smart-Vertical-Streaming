# Disney SD — Smart Vertical Basketball Streaming

Automatically reframes landscape basketball video into vertical 9:16 format using real-time computer vision and object tracking.

## Project Overview

Most sports live streams are produced in the horizontal 16:9 aspect ratio, but mobile audiences prefer the vertical 9:16 format. Naively cropping a horizontal feed often cuts off the player or the ball, causing viewers to miss critical moments.

This project uses YOLOv8 detection and ByteTrack multi-object tracking to intelligently reframe basketball footage so the action stays centered. A Streamlit web UI lets you upload a video and run any of three analysis modes interactively.

## Features

- **Player Detection** — detects and tracks all players on the court using YOLOv8 + ByteTrack, assigning persistent IDs across frames
- **Ball Detection** — locates the basketball each frame using YOLOv8, with a Kalman filter to smooth predictions when the ball is briefly out of view
- **Smart Crop** — automatically reframes the video to 9:16 by following the ball, with smoothed camera movement to avoid jarring cuts

## Project Structure

```
Disney-SD-Smart-Vertical-Streaming/
├── UI/
│   ├── StreamJup.py          # Streamlit app — entry point
│   ├── detection.py          # Processing engine (player, ball, smart crop)
│   ├── html_builders.py      # HTML/JS video players with overlay support
│   └── video_utils.py        # Video encoding and local HTTP server
├── cropping/
│   └── smartCroppingVideos.py  # BallTracker, check_ball(), crop constants
├── detection/
│   ├── ball-detection/
│   │   └── ball_detection.py   # Kalman filter + standalone ball detection script
│   └── player-detection/
│       └── player_detection.py # Standalone player tracking script
├── tests/                    # Unit and integration tests
├── models/                   # YOLOv8 weights (n / s / m)
├── videos/                   # Input test videos (git-ignored)
├── output/                   # Output from standalone scripts
├── run_ui.bat                # One-click UI launcher (Windows)
└── requirements.txt
```

## Setup

**1. Clone the repo**
```bash
git clone <repo-url>
cd Disney-SD-Smart-Vertical-Streaming
```

**2. Create and activate the virtual environment**
```bash
python -m venv sd_disney

# Windows (PowerShell)
.\sd_disney\Scripts\Activate.ps1

# Windows (CMD)
sd_disney\Scripts\activate.bat

# Mac / Linux
source sd_disney/bin/activate
```

**3. Install dependencies**
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
| `opencv-python`   | Video I/O, Kalman filter, H.264 encoding |
| `streamlit`       | Web UI |
| `numpy`           | Array math, ball trajectory fitting |
| `torch`           | CUDA device detection (standalone scripts) |
| `tqdm`            | Progress bars (standalone scripts) |

## Basketball Footage

Test footage is sourced from [Pexels](https://www.pexels.com) (royalty-free).