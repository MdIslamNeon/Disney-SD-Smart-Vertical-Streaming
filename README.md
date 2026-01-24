# Ball Detection & Tracking (YOLOv8 + Kalman Filter)

## Overview
This project implements basketball detection and tracking from video using a GPU-accelerated pipeline. It uses **YOLOv8** for object detection and an **OpenCV Kalman Filter** for smoothing and prediction of ball motion. The system is designed to run **headless** (no GUI) inside **WSL2 with CUDA**, making it compatible with both local NVIDIA GPUs and the remote RTX A4500 workstation.

The output is an annotated video (`output.mp4`) showing:
- Green marker: YOLO detection
- Red marker: Kalman prediction

---

## Technologies Used
- **Python 3.10+**
- **Ultralytics YOLOv8**
- **PyTorch (CUDA-enabled)**
- **OpenCV (headless)**
- **NumPy**
- **WSL2 (Ubuntu)**
- **NVIDIA GPU (RTX 2070 locally, RTX A4500 workstation)**

---

## System Requirements

### Operating System
- Windows 11  
- WSL2 with Ubuntu installed

### GPU
- NVIDIA GPU with CUDA support  
- NVIDIA driver installed on Windows (WSL-compatible)

## Python Environment Setup

1) Create and activate a virtual environment
    - python3 -m venv smart_vertical_env
    - source smart_vertical_env/bin/activate

2) Upgrade pip 
    - pip install --upgrade pip

3) Install dependencies
    - pip install -r requirements.txt

    Note:
     CUDA is provided by the NVIDIA driver and PyTorch CUDA wheels, not via requirements.txt.

## Verify CUDA Support

- Run: python -c "import torch; print(torch.cuda.is_available())"

- Expected output: True

## Running the Code

1) Ensure the video file is inside WSL (home directory)
    - /home/<username>/clip1.mp4

2) Activate the environment
    - source smart_vertical_env/bin/activate

3) Run the script
    - python basketballTest.py

## Output

- The processed video is saved as output.mp4

## Viewing the output

- From Windows File Explorer, open:
     \\wsl$\Ubuntu\home\<username>\

## Common Issues & Fixes

FileNotFoundError: clip1.mp4 does not exist
    - Ensure the video is inside WSL or use a valid /mnt/c/... path.

torch.cuda.is_available() returns False
    - Confirm nvidia-smi works in WSL
    - Ensure PyTorch CUDA wheels are installed
    - Verify NVIDIA driver supports WSL2

Qt / xcb plugin errors
    - Ensure opencv-python-headless is installed
    - Ensure no GUI OpenCV calls remain in the code

## Compatibility with RTX A4500 Workstation

This pipeline mirrors the workstation environment:

    - Linux
    - CUDA
    - Headless execution
    - GPU-accelerated inference

No code changes are required when moving from local RTX GPUs to the RTX A4500 workstation.
 