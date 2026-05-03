"""Retrain both face and smile models with augmented data."""
import torch
import torch.nn as nn
import torch.optim as optim
from config import (
    DEVICE, face_weights_path, smile_weights_path,
    face_epochs, smile_epochs, face_lr, smile_lr,
)
from model import FaceNet, SmileNet
from dataset import get_face_dataloader, get_smile_dataloader
from train import train_model, save_weights


def main():
    # --- Train face model ---
    print("=" * 50)
    print("Training FaceNet...")
    print("=" * 50)
    face_loader = get_face_dataloader(split="train")
    num_face_classes = len(face_loader.dataset.classes)
    print(f"Face classes ({num_face_classes}): {face_loader.dataset.classes}")
    print(f"Training samples: {len(face_loader.dataset)}")

    face_model = FaceNet(num_classes=num_face_classes)
    face_optimizer = optim.Adam(filter(lambda p: p.requires_grad, face_model.parameters()), lr=face_lr)
    face_loss_fn = nn.CrossEntropyLoss()

    train_model(face_model, face_epochs, face_loader, face_loss_fn, face_optimizer)
    save_weights(face_model, face_weights_path)

    # --- Train smile model ---
    print("\n" + "=" * 50)
    print("Training SmileNet...")
    print("=" * 50)
    smile_loader = get_smile_dataloader(split="train")
    print(f"Smile classes: {smile_loader.dataset.classes}")
    print(f"Training samples: {len(smile_loader.dataset)}")

    smile_model = SmileNet()
    smile_optimizer = optim.Adam(filter(lambda p: p.requires_grad, smile_model.parameters()), lr=smile_lr)
    smile_loss_fn = nn.CrossEntropyLoss()

    train_model(smile_model, smile_epochs, smile_loader, smile_loss_fn, smile_optimizer)
    save_weights(smile_model, smile_weights_path)

    # --- Validate ---
    print("\n" + "=" * 50)
    print("Validation Results")
    print("=" * 50)
    face_model.eval()
    smile_model.eval()

    face_val_loader = get_face_dataloader(split="val")
    smile_val_loader = get_smile_dataloader(split="val")

    with torch.no_grad():
        # Face validation
        correct = total = 0
        for batch, labels in face_val_loader:
            batch, labels = batch.to(DEVICE), labels.to(DEVICE)
            preds = face_model(batch).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        print(f"Face validation accuracy: {100.0 * correct / total:.1f}% ({correct}/{total})")

        # Smile validation
        correct = total = 0
        for batch, labels in smile_val_loader:
            batch, labels = batch.to(DEVICE), labels.to(DEVICE)
            preds = smile_model(batch).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        print(f"Smile validation accuracy: {100.0 * correct / total:.1f}% ({correct}/{total})")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()