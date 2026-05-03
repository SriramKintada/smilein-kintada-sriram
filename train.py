import torch
import torch.nn as nn
import torch.optim as optim
from config import DEVICE, face_weights_path, smile_weights_path, checkpoints_dir
import os


def train_model(model, num_epochs, train_loader, loss_fn, optimizer):
    model.to(DEVICE)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch, labels in train_loader:
            batch = batch.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.1f}%")

    return model


def train_face_model(model, num_epochs, train_loader, loss_fn, optimizer):
    return train_model(model.face_net, num_epochs, train_loader, loss_fn, optimizer)


def train_smile_model(model, num_epochs, train_loader, loss_fn, optimizer):
    return train_model(model.smile_net, num_epochs, train_loader, loss_fn, optimizer)


def save_weights(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved weights to {path}")