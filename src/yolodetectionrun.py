from ultralytics import YOLO

# Load a pre-trained YOLOv8s model
model = YOLO('yolov8s.pt')

# grab video folder
video_folder = '/content/ultralytics/ultralytics/videos'
video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))] # Add other video extensions if needed and get full paths

# incase video file is empty...
if not video_files:
    print(f"No video files found in {video_folder}")
else:
    print(f"Found {len(video_files)} video files to process.")
    # Process each video file individually and save results to a single folder
    # iterates thru videos. (for now, leave conf at 0.3)
    for video_index, video_path in enumerate(video_files):
        #print(f"Processing video {video_index + 1}/{len(video_files)}: {os.path.basename(video_path)}")
        # run the model on the current video
        results = model.predict(video_path, classes=[0], stream=False, save=True, conf=0.3, project='runs/detect', name='single_output_folder') # classes=[0] for 'person'

        # Iterate thru results (one result object per frame) and summarize
        for i, result in enumerate(results):
            # Access the detections for the current frame
            detections = result.boxes

            # count the number of detections (classes=[0] = people)
            num_people = len(detections)
            print(f"  Video {video_index + 1}, Frame {i}: Detected {num_people} people")
