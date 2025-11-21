from ultralytics import YOLO

# Load models
detector = YOLO("models/detector.pt")
classifier = YOLO("models/classifier.pt")

def pipeline(image_path):
    print("\n[INFO] Running detection on:", image_path)

    # -------------------------
    # 1. Run DETECTION
    # -------------------------
    det_result = detector(image_path, save=True)

    if len(det_result[0].boxes) == 0:
        print("[DETECTION] No regions detected → sending full image to classifier\n")
        crop_path = image_path
    else:
        print("[DETECTION] Object detected → cropping first bounding box")
        box = det_result[0].boxes.xyxy[0].tolist()

        import cv2
        img = cv2.imread(image_path)
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]

        crop_path = "cropped_temp.jpg"
        cv2.imwrite(crop_path, crop)

    # -------------------------
    # 2. Run CLASSIFICATION
    # -------------------------
    print("[INFO] Running classification...\n")

    pred = classifier(crop_path, save=True)

    # Extract probabilities
    probs = pred[0].probs.data.tolist()

    fake_score = probs[0]
    real_score = probs[1]

    print("===== FINAL RESULT =====")
    if real_score > fake_score:
        print(f"REAL DOCUMENT ✔ (confidence: {real_score:.2f})")
    else:
        print(f"FAKE DOCUMENT ✖ (confidence: {fake_score:.2f})")


# Run pipeline
test_image = input("Enter image path: ")
pipeline(test_image)
