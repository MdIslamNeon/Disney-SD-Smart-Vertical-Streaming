from ultralytics import YOLO

# Load a pre-trained YOLOv8s model
model = YOLO('yolov8s.pt')

# run the model on selected video:
# if multiple videos, consider running thru a list (for now, leave conf at 0.3 cause those 2 people wrestling at the middle wont be marked...)
results = model.predict('testbsk.mp4', classes=[0], stream=True, save=True, conf=0.3) # classes=[0] for 'person'

# Iterate thru results (one result object per frame) and summarize
for i, result in enumerate(results):
    # Access the detections for the current frame
    detections = result.boxes

    # count the number of detections (classes=[0] = people)
    num_people = len(detections)
    print(f"Frame {i}: Detected {num_people} people")