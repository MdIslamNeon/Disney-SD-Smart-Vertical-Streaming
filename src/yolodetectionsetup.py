!git clone https://github.com/ultralytics/ultralytics.git

import ultralytics
import os

# Download dependencies from dependencies txt file in origin repo
dependencies = [
    "numpy",
    "matplotlib",
    "opencv-python",
    "pillow",
    "pyyaml",
    "requests",
    "scipy",
    "torch",
    "torchvision",
    "psutil",
    "polars",
    "ultralytics-thop",
]

for dep in dependencies:
    print(f"Checking {dep}:")
    !pip show {dep}

    print("-" * 20)

# create videos folder to store data. run this before inputting files for now.
folder_path = '/content/ultralytics/ultralytics/videos'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")
else:
    print(f"Folder '{folder_path}' already exists.")
