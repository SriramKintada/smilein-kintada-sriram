"""Quick script to classify sriram's webcam photos into smiling/not_smiling using OpenCV."""
import cv2
import os
import shutil

src_dir = r"data\face_identity\sriram"
smile_dir = r"data\smile\smiling"
nosmile_dir = r"data\smile\not_smiling"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

smiling_files = []
not_smiling_files = []

for fname in sorted(os.listdir(src_dir)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img_path = os.path.join(src_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    has_smile = False
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.3, minNeighbors=15, minSize=(20, 20))
        if len(smiles) > 0:
            has_smile = True
            break

    if has_smile:
        smiling_files.append(fname)
    else:
        not_smiling_files.append(fname)

print(f"Smiling: {len(smiling_files)} photos")
print(f"Not smiling: {len(not_smiling_files)} photos")
print(f"\nSmiling files: {smiling_files}")

# Copy to smile data directories with sriram_ prefix to avoid name collisions
os.makedirs(smile_dir, exist_ok=True)
os.makedirs(nosmile_dir, exist_ok=True)

for fname in smiling_files:
    dst = os.path.join(smile_dir, f"sriram_{fname}")
    shutil.copy2(os.path.join(src_dir, fname), dst)

for fname in not_smiling_files:
    dst = os.path.join(nosmile_dir, f"sriram_{fname}")
    shutil.copy2(os.path.join(src_dir, fname), dst)

print(f"\nCopied {len(smiling_files)} to {smile_dir}")
print(f"Copied {len(not_smiling_files)} to {nosmile_dir}")