import cv2
from ultralytics import YOLO

def main():
    mp4_file = "ft0_v108_002649_x264.mp4"
    # Load YOLOv8n (nano, fast, lightweight)
    model = YOLO("yolov8n.pt")


    results = model.predict(
        source=mp4_file,
        stream=True,
        save=True,
    )

    window_name = "YOLO Detections"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)  # Set to your preferred size

    for result in results:
        frame = result.plot()
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
