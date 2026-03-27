import cv2                      
import os
from pathlib import Path
import kagglehub                


DATASET_ID = "sarbagyashakya/basketball-51-dataset"
OUT_W, OUT_H = 540, 960         
         

dataset_path = Path(kagglehub.dataset_download(DATASET_ID))
print("Dataset folder:", dataset_path)

# Create an output folder next to the dataset for cropped videos.
output_root = dataset_path.parent / (dataset_path.name + "_vertical")
output_root.mkdir(exist_ok=True)
print("Output folder:", output_root)

# Walk through the dataset folder and collect ALL .mp4 files.
video_files = []
for root, _, files in os.walk(dataset_path):
    for f in files:
        if f.lower().endswith(".mp4"):                 
            full_path = Path(root) / f                
            video_files.append(full_path)

print(f"Found {len(video_files)} videos.")

processed = 0

# 4) Loop over EVERY video in the dataset (needed to process all clips).
for idx, video_path in enumerate(video_files, start=1):

   
    #Creates a VideoCapture object to read frames.
    cap = cv2.VideoCapture(str(video_path))

    #Checks if the video file was opened successfully.
    if not cap.isOpened():
        print("Error: Could not open video:", video_path)
        continue

    # Read one frame to get properties (size, fps).
    ret, frame = cap.read()  
    fps = cap.get(cv2.CAP_PROP_FPS) or 30                            
    if not ret:
        print("Warning: empty/broken video:", video_path)
        cap.release()
        continue

    # Get frame size (width, height) and FPS from the capture.
    h, w = frame.shape[:2]


 
    # For your 320x240 clips: keep full height (240) and crop width to 240*(9/16)=135, centered.
    target_w = int(round(h * 9 / 16))                  
    target_w = min(target_w, w)                         
    x1 = (w - target_w) // 2                          
    x2 = x1 + target_w                                
    # cropped = frame[:, x1:x2] will keep all rows (0..h) and only columns x1..x2


    # We mirror the original relative path inside output_root to keep class labels.
    rel_path = video_path.relative_to(dataset_path)     # e.g., "2p0/clip123.mp4"
    out_path = (output_root / rel_path).with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure class folder exists in output

    # fourcc + VideoWriter: where we will write the cropped frames (resized to OUT_W x OUT_H)
    
    #! Error here fix later! !#
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (OUT_W, OUT_H))

    # Write the FIRST frame we already read, then continue with the rest in a loop.
    def write_cropped(f):
        cropped = f[:, x1:x2]                          
        resized = cv2.resize(cropped, (OUT_W, OUT_H))   
        writer.write(resized)   
        return True                       

        

    #Write the first frame we already have
    if not write_cropped(frame):
        cap.release()
        writer.release()
        continue

    # while True: Enters a loop to read frames continuously until the video ends.
    while True:
        ret, frame = cap.read()                         
        if not ret:                                     
            break
        if not write_cropped(frame):                   
            break

    # cap.release(): Releases the VideoCapture object (frees the file)
    cap.release()
    # writer.release(): Finalize and close the output video file
    writer.release()

    processed += 1
    print("Saved to:", out_path)

# cv2.destroyAllWindows(): Closes all OpenCV windows (if we showed any)
cv2.destroyAllWindows()
print(f"\nDone! Processed {processed}/{len(video_files)} videos.")
print("All cropped videos are in:", output_root)
