# SmileIn: Smile-Verified Face Recognition Attendance System

A dual-CNN system that recognizes faces and detects smiles via webcam. Attendance is logged only when a recognized person smiles, providing liveness verification and intent signaling.

**Author:** Kintada Sriram (ID: 20231119)  
**Course:** Image and Video Processing

## Project Structure

```
project_kintada_sriram/
├── checkpoints/           # Saved model weights
│   ├── face_net_weights.pth
│   └── smile_net_weights.pth
├── data/                  # Training data
│   ├── face_identity/     # One subfolder per enrolled person
│   │   ├── ben_afflek/
│   │   ├── elton_john/
│   │   ├── jerry_seinfeld/
│   │   ├── madonna/
│   │   ├── mindy_kaling/
│   │   └── sriram/        # Self-collected webcam photos
│   └── smile/
│       ├── smiling/       # ~649 images (GENKI-4K + sriram webcam)
│       └── not_smiling/   # ~611 images (GENKI-4K + sriram webcam)
├── config.py              # All hyperparameters and paths
├── dataset.py             # FaceIdentityDataset, SmileDataset, dataloaders
├── model.py               # FaceNet, SmileNet, SmileInModel
├── train.py               # train_model() and save_weights()
├── predict.py             # predict_identity(), predict_smile(), predict_attendance()
├── interface.py           # Standardized imports for grading
├── enroll.py              # Webcam enrollment script
├── live_demo.py           # Real-time webcam attendance demo
└── README.md
```

## Installation

```bash
pip install torch torchvision opencv-python pillow
```

## Training

```bash
python retrain.py
```

This trains both FaceNet (20 epochs) and SmileNet (10 epochs) and saves weights to `checkpoints/`.

Results (after training):
- FaceNet validation accuracy: ~98%
- SmileNet validation accuracy: ~85%

## Prediction

### Single/batch prediction (for grading)
```python
from predict import predict_attendance
results = predict_attendance(["data/face_identity/sriram/img0001.jpg"])
# Returns: ["sriram (smiling, attendance logged)"]
```

### Identity only
```python
from predict import predict_identity
results = predict_identity(["path/to/image.jpg"])
# Returns: [("sriram", 0.77)]
```

### Smile only
```python
from predict import predict_smile
results = predict_smile(["path/to/image.jpg"])
# Returns: [(True, 0.85)]
```

## Live Demo

```bash
python live_demo.py
```

Opens your webcam. The system detects faces, recognizes identity, and checks for smiles. A person's name appears only when they smile. After smiling for 3 consecutive frames, attendance is logged to `attendance.csv`.

Press `q` to quit.

## Enrollment

To add a new person:
```bash
python enroll.py
```

Enter the person's name and press SPACE to capture photos. Collect ~30 photos with varied angles and lighting. Then retrain both models.

## Grading Interface

The `interface.py` file standardizes all imports:

| Grading Name           | Maps To                 | Source File  |
|------------------------|-------------------------|--------------|
| TheModel               | SmileInModel            | model.py     |
| TheFaceNet             | FaceNet                 | model.py     |
| TheSmileNet            | SmileNet                | model.py     |
| the_trainer            | train_model             | train.py     |
| the_predictor          | predict_attendance      | predict.py   |
| TheDataset             | FaceIdentityDataset     | dataset.py   |
| the_dataloader         | get_face_dataloader     | dataset.py   |
| the_batch_size         | face_batch_size (16)    | config.py    |
| total_epochs           | face_epochs (20)        | config.py    |

## Architecture

Both CNNs use pretrained MobileNetV2 (ImageNet) with transfer learning:

- **FaceNet**: MobileNetV2 backbone with features[:14] frozen, final layer replaced with Linear(1280, 6). Trained on 6 classes (5 celebrities + sriram). Input: 224x224 RGB.
- **SmileNet**: MobileNetV2 backbone with features[:10] frozen, final layer replaced with Linear(1280, 2). Binary classification (smiling/not_smiling). Input: 224x224 RGB.
- **Face Detection**: OpenCV Haar Cascade (haarcascade_frontalface_default.xml) with 30% margin crop.
- **Temporal Smoothing**: 3 consecutive smile frames required before logging attendance.
- **Unknown Threshold**: Face identity confidence below 0.5 is classified as "Unknown".

## Datasets

- **Face Identity**: 6 classes (ben_afflek, elton_john, jerry_seinfeld, madonna, mindy_kaling, sriram). ~19-55 images per class, total 173 images. Celeb photos from VGGFace2, sriram photos self-collected via webcam. 70/30 train/val split.
- **Smile Detection**: GENKI-4K dataset (~1200 images) plus 55 sriram webcam photos classified by OpenCV Haar cascade smile detector. Total ~1260 images. 70/30 train/val split.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenCV (cv2)
- Pillow