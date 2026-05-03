import cv2
import os
from config import face_data_dir


def enroll():
    name = input("Enter your name (no spaces, use underscore): ").strip()
    person_dir = os.path.join(face_data_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    count = 0
    target = 30
    print(f"Capturing {target} photos for '{name}'. Move your head around for varied angles.")
    print("Press 'q' to quit early.")

    while count < target:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Enrollment - Press SPACE to capture, Q to quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            filepath = os.path.join(person_dir, f"img{count+1:04d}.jpg")
            cv2.imwrite(filepath, frame)
            count += 1
            print(f"  Captured {count}/{target}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved {count} images to {person_dir}")


if __name__ == "__main__":
    enroll()