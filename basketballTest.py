import cv2
import torch #future GPU implementation 
from ultralytics import YOLO


def main():
    #mp4_file = "ft0_v108_002649_x264.mp4"
    #mp4_file = "1080p60fps_RT.mp4"
    #mp4_file = "1080p60fps_RT2.mp4"
    #mp4_file = "3840_2160_30fps_RT3.mp4"
    #mp4_file = "3840_2160_30fps_RT4.mp4"
    mp4_file = "1920_1080_30fps_RT5.mp4"

    # Load YOLOv8n (COCO pretrained)
    model = YOLO("yolov8m.pt")

    # Get ID for 'sports ball'
    ball_class_id = [k for k, v in model.names.items() if v == "sports ball"][0]
    #person_class_id = [k for k, v in model.names.items() if v == "person"][0] #remove comment if necessary 
    print(f"Detected 'sports ball' class ID: {ball_class_id}")

    # Run detection stream
    results = model.predict(
        source=mp4_file,
        tracker="bytetrack.yaml", #redudancy, byterack should be on by default
        stream=True,
        imgsz=992, #default 640 - multiples of 32. (tips: 640 = 60fps, 1280 + 0.10conf -> 30fps)
        conf=0.40, #default 0.25 (tips: 0.10 for nano #0.4 for small - disregard for now 10)
        iou = 0.4, #overlapping - lower is less overlap
        save=True,
        classes=[ball_class_id,], #only the designated sports ball will be detected, add person_class_id if necessary
        
        
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
