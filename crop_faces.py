"""Crop sriram's webcam photos to face regions to match celeb training data dimensions."""
import cv2
import os
from PIL import Image

src_dir = r"data\face_identity\sriram"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cropped = 0
skipped = 0

for fname in sorted(os.listdir(src_dir)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img_path = os.path.join(src_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(80, 80))

    if len(faces) == 0:
        skipped += 1
        os.remove(img_path)
        continue

    # Take the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # Add 30% margin
    margin_x = int(w * 0.3)
    margin_y = int(h * 0.3)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img.shape[1], x + w + margin_x)
    y2 = min(img.shape[0], y + h + margin_y)

    crop = img[y1:y2, x1:x2]
    cv2.imwrite(img_path, crop)
    cropped += 1

print(f"Cropped: {cropped}, Removed (no face): {skipped}")