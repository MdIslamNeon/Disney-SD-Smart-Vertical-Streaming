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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def kalman_filter():
    # 4 state variables (x, y, vx, vy) and 2 coordinates (x, y) for YOLO
    kf = cv2.KalmanFilter(4, 2)

    # Measure x and y coordinates only from the 4 state variables (x, y, vx, vy)
    # YOLO only knows x and y and not velocity
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
    )

    # This defines how the object moves from one frame to the next
    # Each frame, the ball's position changes based on its velocity
    #
    # x_next = x + vx (1*x + 0*y + 1*vx + 0*vy) # <- 1st array
    # y_next = y + vy (0*x + 1*y + 0*vx + 1*vy) # <- 2nd array
    # vx_next = vx   (0*x + 0*y + 1*vx + 0*vy) # <- 3rd array
    # vy_next = vy   (0*x + 0*y + 0*vx + 1*vy) # <- 4th array
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1],
         [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
    )

    # This defines how much random movement or uncertainty we allow in the model
    # A higher value -> more "flexible" predictions (follows the object faster but noisier)
    # A lower value -> smoother predictions (slower to respond to quick movement)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.06     # more responsive
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.3  # trust YOLO more
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf


def main():

    BASE_DIR = Path(__file__).resolve().parent.parent
    mp4_file = BASE_DIR / "videos" / "testing_video.mp4"
    out_path  = BASE_DIR / "output" / f"{mp4_file.stem}_ball_detection.mp4"

    model = YOLO("../models/yolov8m.pt")

    # Get ID for 'sports ball'
    ball_class_id = [k for k, v in model.names.items() if v == "sports ball"][0]
    #person_class_id = [k for k, v in model.names.items() if v == "person"][0]
    print(f"Detected 'sports ball' class ID: {ball_class_id}")

    # Run detection stream
    results = model.predict(
        source=str(mp4_file),
        #tracker="bytetrack.yaml",
        stream=True,
        imgsz=768,  #default 640 - multiples of 32. (tips: 640 = 60fps, 1280 + 0.10conf -> 30fps)
        conf=0.20,  #default 0.25 (tips: 0.10 for nano #0.4 for small - disregard for now 10)
        iou=0.4,    #overlapping - lower is less overlap
        save=False,
        classes=[ball_class_id],
        device=DEVICE,
    )

    window_name = "YOLO Detections"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)

    # Initialize Kalman and FPS counter
    kf = kalman_filter()
    frame_i = 0
    t0 = time.time()
    out = None

    for result in results:
        frame = result.orig_img  # use raw frame for better FPS

        # Initialize video writer ONCE
        if out is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            out = cv2.VideoWriter(str(out_path), fourcc, 30, (w, h))

        pred = kf.predict()      # always predict first
        px, py = int(pred[0]), int(pred[1])
        have_det = False

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            confs = boxes.conf.cpu().numpy() # type: ignore
            classes = boxes.cls.cpu().numpy() # type: ignore

            # find highest confidence sports ball
            idx = np.argmax(confs)
            if int(classes[idx]) == ball_class_id:
                xyxy = boxes.xyxy[idx].cpu().numpy()

                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
                measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
                kf.correct(measurement)
                have_det = True

                # Hard-reset if YOLO is confident
                if confs[idx] > 0.6:
                    kf.statePre[:2] = measurement
                    kf.statePost[:2] = measurement

        # Draw YOLO (green) and Kalman (red)
        if have_det:
            cv2.circle(frame, (int(cx), int(cy)), 8, (0, 255, 0), -1)  # green YOLO

        cv2.circle(frame, (px, py), 8, (0, 0, 255), 2)                  # red Kalman

        label = "YOLO Detected" if have_det else "Kalman Predict"
        color = (0, 255, 0) if have_det else (0, 0, 255)
        cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

        # FPS display
        frame_i += 1
        if frame_i % 30 == 0:
            fps = frame_i / (time.time() - t0)
            print(f"[PERF] {fps:.2f} FPS")

        out.write(frame)

    if out is not None:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()