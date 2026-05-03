import cv2
import torch
import os
import csv
from datetime import datetime
from facenet_pytorch import MTCNN
from torchvision import transforms
from config import (
    DEVICE, resize_x, resize_y,
    imagenet_mean, imagenet_std,
    unknown_threshold, smile_frames_required,
    face_weights_path, smile_weights_path,
    face_data_dir, attendance_csv,
)
from model import FaceNet, SmileNet


def run_live_demo():
    classes = sorted([
        d for d in os.listdir(face_data_dir)
        if os.path.isdir(os.path.join(face_data_dir, d))
    ]) if os.path.isdir(face_data_dir) else []

    face_model = FaceNet(num_classes=len(classes))
    smile_model = SmileNet()

    if os.path.exists(face_weights_path):
        face_model.load_state_dict(torch.load(face_weights_path, map_location=DEVICE, weights_only=True))
    if os.path.exists(smile_weights_path):
        smile_model.load_state_dict(torch.load(smile_weights_path, map_location=DEVICE, weights_only=True))

    face_model.to(DEVICE).eval()
    smile_model.to(DEVICE).eval()

    mtcnn = MTCNN(image_size=160, margin=20, keep_confidence=True, device=DEVICE)

    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    smile_counter = {}
    logged = set()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("SmileIn live demo running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(rgb_frame)

        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob is None or prob < 0.9:
                    continue

                x1, y1, x2, y2 = map(int, box)
                face_crop = rgb_frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                face_pil = transforms.functional.to_pil_image(face_crop)
                face_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    face_logits = face_model(face_tensor)
                    face_probs = torch.softmax(face_logits, dim=1)
                    face_conf, face_pred = face_probs.max(dim=1)

                    smile_logits = smile_model(face_tensor)
                    smile_probs = torch.softmax(smile_logits, dim=1)
                    smile_conf = smile_probs[0, 1].item()
                    is_smiling = smile_conf > 0.5

                name = classes[face_pred.item()] if face_conf.item() >= unknown_threshold else "Unknown"

                if name != "Unknown":
                    key = name
                    if is_smiling:
                        smile_counter[key] = smile_counter.get(key, 0) + 1
                    else:
                        smile_counter[key] = 0

                    if smile_counter.get(key, 0) >= smile_frames_required and key not in logged:
                        logged.add(key)
                        with open(attendance_csv, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([key, datetime.now().strftime("%Y-%m-%d"),
                                            datetime.now().strftime("%H:%M:%S"),
                                            round(face_conf.item(), 4),
                                            round(smile_conf, 4)])
                        print(f"  Logged attendance: {key}")

                color = (0, 255, 0) if is_smiling and name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = name if is_smiling and name != "Unknown" else "..."
                smile_tag = "Smiling" if is_smiling else "Not Smiling"
                cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, smile_tag, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("SmileIn - Smile to Check In (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Attendance saved to {attendance_csv}")
    print(f"Checked in: {list(logged)}")


if __name__ == "__main__":
    run_live_demo()