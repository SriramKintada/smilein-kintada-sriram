# SmileIn: Smile-Verified Face Recognition Attendance System

A dual-CNN system that recognizes faces and detects smiles via webcam. Attendance is logged only when a recognized person smiles, providing liveness/anti-spoofing and an intent signal.

## Project Structure

```
project_kintada_sriram/
├── checkpoints/           # Saved model weights
│   ├── face_net_weights.pth
│   └── smile_net_weights.pth
├── data/                  # Sample data (10 items per class, unresized)
│   ├── face_identity/     # One subfolder per enrolled person
│   │   ├── person1/       # img0001.jpg ... img0010.jpg
│   │   ├── person2/
│   │   └── ...
│   └── smile/
│       ├── smiling/       # 10 smiling face images
│       └── not_smiling/   # 10 non-smiling face images
├── config.py              # All hyperparameters and paths
├── dataset.py             # FaceIdentityDataset, SmileDataset, dataloaders
├── model.py               # FaceNet, SmileNet, SmileInModel (wraps both)
├── train.py                # train_model() and save_weights()
├── predict.py              # predict_identity(), predict_smile(), predict_attendance()
├── interface.py            # Standardized imports for grading
├── enroll.py               # Webcam enrollment script
├── live_demo.py            # Real-time webcam attendance demo
└── README.md
```

## Installation

```bash
pip install torch torchvision opencv-python facenet-pytorch pillow pandas
```

## Data Setup

### Face Identity Data
Each enrolled person has a subfolder inside `data/face_identity/` containing their photos. To collect enrollment photos:

```bash
python enroll.py
```

This opens your webcam. Enter your name, then press SPACE to capture each photo. Collect ~30 photos per person with varied angles and lighting.

### Smile Data
Download the GENKI-4K dataset and place 10 smiling images in `data/smile/smiling/` and 10 non-smiling images in `data/smile/not_smiling/`. For full training, place the entire dataset elsewhere and update `smile_data_dir` in `config.py`.

## Training

### Train the Face Identity CNN
```python
from dataset import get_face_dataloader
from model import FaceNet
from train import train_model, save_weights
from config import face_epochs, face_lr, face_weights_path
import torch.nn as nn
import torch.optim as optim
import os

num_classes = len(os.listdir("data/face_identity"))
model = FaceNet(num_classes=num_classes)
train_loader = get_face_dataloader(split="train")
optimizer = optim.Adam(model.parameters(), lr=face_lr)
loss_fn = nn.CrossEntropyLoss()

train_model(model, face_epochs, train_loader, loss_fn, optimizer)
save_weights(model, face_weights_path)
```

### Train the Smile CNN
```python
from dataset import get_smile_dataloader
from model import SmileNet
from train import train_model, save_weights
from config import smile_epochs, smile_lr, smile_weights_path
import torch.nn as nn
import torch.optim as optim

model = SmileNet()
train_loader = get_smile_dataloader(split="train")
optimizer = optim.Adam(model.parameters(), lr=smile_lr)
loss_fn = nn.CrossEntropyLoss()

train_model(model, smile_epochs, train_loader, loss_fn, optimizer)
save_weights(model, smile_weights_path)
```

## Prediction

### Predict on image paths (for grading)
```python
from predict import predict_attendance
results = predict_attendance(["data/face_identity/person1/img0001.jpg"])
print(results)
# [{'name': 'person1', 'identity_confidence': 0.97, 'smiling': True, 'smile_confidence': 0.88, 'attendance': True}]
```

### Predict identity only
```python
from predict import predict_identity
results = predict_identity(["path/to/image.jpg"])
print(results)  # [('person1', 0.97)]
```

### Predict smile only
```python
from predict import predict_smile
results = predict_smile(["path/to/image.jpg"])
print(results)  # [(True, 0.88)]
```

## Live Demo

```bash
python live_demo.py
```

Opens your webcam. The system detects faces, recognizes identity, and checks for smiles. A person's name appears only when they smile. After smiling for 3 consecutive frames, attendance is logged to `attendance.csv`.

Press `q` to quit.

## Grading Interface

The `interface.py` file standardizes all imports for automated grading:

| Grading Name         | Maps To                  | Source File  |
|----------------------|--------------------------|--------------|
| TheModel            | SmileInModel             | model.py     |
| TheFaceNet          | FaceNet                  | model.py     |
| TheSmileNet         | SmileNet                 | model.py     |
| the_trainer         | train_model              | train.py     |
| the_predictor       | predict_attendance       | predict.py   |
| the_identity_predictor | predict_identity      | predict.py   |
| the_smile_predictor | predict_smile            | predict.py   |
| TheDataset          | FaceIdentityDataset      | dataset.py   |
| TheSmileDataset     | SmileDataset             | dataset.py   |
| the_dataloader      | get_face_dataloader      | dataset.py   |
| the_smile_dataloader | get_smile_dataloader   | dataset.py   |
| the_batch_size      | face_batch_size (16)     | config.py    |
| total_epochs        | face_epochs (20)         | config.py    |

## Architecture

Both CNNs use pretrained MobileNetV2 (ImageNet) with transfer learning:

- **Face Identity CNN**: MobileNetV2 backbone, final layer replaced with Linear(in, N_classes) where N = number of enrolled people. Fine-tuned on self-collected enrollment photos.
- **Smile CNN**: MobileNetV2 backbone, final layer replaced with Linear(in, 2) for smiling/not-smiling. Fine-tuned on GENKI-4K dataset.
- **Face Detection**: MTCNN from facenet-pytorch (pretrained, used only for cropping faces in the live demo).
- **Temporal Smoothing**: A person must smile for 3 consecutive video frames before attendance is logged.

## Datasets

- **Face Identity**: Self-collected from laptop webcam. ~30 photos per person, 8-10 participants. Split 70/15/15 train/val/test.
- **Smile Detection**: GENKI-4K (public, ~4,000 labeled face images). Backup: CelebA "Smiling" attribute.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenCV (cv2)
- facenet-pytorch
- Pillow