import os
import cv2

# Input and output folders
INPUT_DIR = "data/raw/aadhar_dataset/test/images/"
OUTPUT_DIR = "data/preprocessed/"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process each image
for file in os.listdir(INPUT_DIR):
    if file.endswith(".jpg") or file.endswith(".png"):
        img_path = os.path.join(INPUT_DIR, file)
        img = cv2.imread(img_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Remove noise
        denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
        
        # Thresholding (binarization)
        _, thresh = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)
        
        # Save preprocessed image
        cv2.imwrite(os.path.join(OUTPUT_DIR, file), thresh)

print("Preprocessing complete! Preprocessed images saved in 'data/preprocessed/'")
