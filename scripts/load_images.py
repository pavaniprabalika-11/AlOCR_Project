import os
from PIL import Image

# Correct path to your Aadhaar images
DATA_DIR = "data/raw/aadhar_dataset/test/images/"

# List all image files
images = [f for f in os.listdir(DATA_DIR) if f.endswith(".jpg") or f.endswith(".png")]
print(f"Total images found: {len(images)}")

# Load and display the first image
if len(images) > 0:
    first_image_path = os.path.join(DATA_DIR, images[0])
    img = Image.open(first_image_path)
    img.show()
else:
    print("No images found in the folder.")

