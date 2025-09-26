!git clone https://github.com/ultralytics/ultralytics.git

import ultralytics

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