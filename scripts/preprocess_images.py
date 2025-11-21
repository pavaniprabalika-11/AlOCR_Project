import cv2
import os
from pathlib import Path

# Input and output folders
input_folder = Path("data/raw")          # folder with original Aadhaar images
output_folder = Path("data/preprocessed")
output_folder.mkdir(parents=True, exist_ok=True)

def preprocess_image(image_path, output_path):
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Skipping {image_path}, not a valid image.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur (reduces small noise)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Use Adaptive Thresholding for better contrast
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # Optional: morphological opening to clean extra dots
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Save preprocessed image
    cv2.imwrite(str(output_path), cleaned)
    print(f"Processed and saved: {output_path}")

def preprocess_all():
    for image_file in input_folder.glob("*.*"):
        if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            output_path = output_folder / image_file.name
            preprocess_image(image_file, output_path)

if __name__ == "__main__":
    preprocess_all()
    print("✅ Image preprocessing completed successfully.")
