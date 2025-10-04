from ultralytics import YOLO
import cv2

# Direct path to video file
video_file = "/Users/an.k.chen/Downloads/2p0_v108_002025_x264.mp4"

print("Path to dataset file:", video_file)

# Load a pretrained YOLO model
model = YOLO("yolov8n.pt")

# Run tracking
results = model.track(source=video_file, show=True, save=True)

# Manual visualization
for result in results:
    frame = result.plot()
    cv2.imshow("YOLO Tracking", frame)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()