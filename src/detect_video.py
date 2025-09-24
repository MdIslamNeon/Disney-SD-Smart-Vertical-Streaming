from ultralytics import YOLO
import os

MODEL_PATH = "yolov8s.pt"                   # pre-trained YOLOv8 model (using small model for speed + accuracy)
SOURCE = "data/basketball_clip.mp4"         # input video (can also be an image)
SAVE_DIR = "outputs/detect"                 # where results will be saved

def main():
    # Make sure save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load YOLOv8 model
    model = YOLO(MODEL_PATH)

    # Run detection
    results = model.predict(
        source=SOURCE,
        save=True,                # saves annotated video
        project=SAVE_DIR,         # base folder for results
        name="run",               # subfolder name
        exist_ok=True,            # overwrite previous "run"
        show=False                 # Don't preview while running
    )

    print(f"\n Detection complete. Results saved to: {SAVE_DIR}/run")

if __name__ == "__main__":
    main()