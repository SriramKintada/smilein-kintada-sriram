import os
import torch
from PIL import Image
from torchvision import transforms
from config import (
    DEVICE, resize_x, resize_y,
    imagenet_mean, imagenet_std,
    unknown_threshold,
    face_weights_path, smile_weights_path,
    face_data_dir,
)
from model import SmileInModel, FaceNet, SmileNet


infer_transform = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])


def _load_image(path):
    img = Image.open(path).convert("RGB")
    return infer_transform(img).unsqueeze(0)


def _load_batch(list_of_img_paths):
    tensors = [_load_image(p) for p in list_of_img_paths]
    return torch.cat(tensors, dim=0).to(DEVICE)


def _get_face_classes():
    if not os.path.isdir(face_data_dir):
        return []
    return sorted([
        d for d in os.listdir(face_data_dir)
        if os.path.isdir(os.path.join(face_data_dir, d))
    ])


def _load_model(model, path, num_face_classes):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


def predict_identity(list_of_img_paths):
    classes = _get_face_classes()
    model = FaceNet(num_classes=len(classes))
    _load_model(model, face_weights_path, len(classes))
    model.eval()

    batch = _load_batch(list_of_img_paths)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)

    results = []
    for pred, conf in zip(preds, confs):
        if conf.item() < unknown_threshold:
            results.append(("Unknown", conf.item()))
        else:
            results.append((classes[pred.item()], conf.item()))
    return results


def predict_smile(list_of_img_paths):
    model = SmileNet()
    _load_model(model, smile_weights_path, 0)
    model.eval()

    batch = _load_batch(list_of_img_paths)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        smile_confs = probs[:, 1]
        smile_preds = (smile_confs > 0.5).long()

    results = []
    for pred, conf in zip(smile_preds, smile_confs):
        results.append((bool(pred.item()), conf.item()))
    return results


def predict_attendance(list_of_img_paths):
    identities = predict_identity(list_of_img_paths)
    smiles = predict_smile(list_of_img_paths)

    results = []
    for (name, id_conf), (smiling, smile_conf) in zip(identities, smiles):
        if name == "Unknown":
            attendance = False
        elif not smiling:
            attendance = False
        else:
            attendance = True
        results.append({
            "name": name,
            "identity_confidence": round(id_conf, 4),
            "smiling": smiling,
            "smile_confidence": round(smile_conf, 4),
            "attendance": attendance,
        })
    return results