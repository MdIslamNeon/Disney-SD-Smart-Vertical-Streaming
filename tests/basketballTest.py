import cv2
import numpy as np
from ultralytics import YOLO

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
        [[1, 0, 1, 0],[0, 1, 0, 1],
         [0, 0, 1, 0],[0, 0, 0, 1]], np.float32
    )

    # This defines how much random movement or uncertainty we allow in the model
    # A higher value -> more "flexible" predictions (follows the object faster but noisier)
    # A lower value -> smoother predictions (slower to respond to quick movement)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kf

def main():

    # Put these videos under the local_videos folder

    #mp4_file = "../ft0_v108_002649_x264.mp4"
    #mp4_file = "1080p60fps_RT.mp4"
    #mp4_file = "1080p60fps_RT2.mp4"
    #mp4_file = "3840_2160_30fps_RT3.mp4"
    #mp4_file = "3840_2160_30fps_RT4.mp4"
    mp4_file = "1920_1080_30fps_RT5.mp4"

    model = YOLO("../models/yolov8m.pt")

    # Get ID for 'sports ball'
    ball_class_id = [k for k, v in model.names.items() if v == "sports ball"][0]
    #person_class_id = [k for k, v in model.names.items() if v == "person"][0]
    print(f"Detected 'sports ball' class ID: {ball_class_id}")

    # Run detection stream
    results = model.track(
        source=mp4_file,
        tracker="bytetrack.yaml",
        stream=True,
        imgsz=992,
        conf=0.40,
        iou=0.4,
        save=True,
        classes=[ball_class_id],
    )

    window_name = "YOLO Detections"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)

    for result in results:
        frame = result.plot()
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
