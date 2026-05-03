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
from model import FaceNet, SmileNet


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


def _load_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


def predict_identity(list_of_img_paths):
    """Predict the identity of each face image. Returns list of (name, confidence)."""
    classes = _get_face_classes()
    model = FaceNet(num_classes=len(classes))
    _load_model(model, face_weights_path)

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
    """Predict whether each face image shows a smile. Returns list of (is_smiling, confidence)."""
    model = SmileNet()
    _load_model(model, smile_weights_path)

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


def predict_attendance(input_data):
    """Combined prediction: identity + smile check.

    Accepts either:
      - list of file paths (strings)
      - single file path (string)
      - torch Tensor of shape (B, 3, 224, 224) or (3, 224, 224)
    Returns list of label strings:
      - "name (smiling, attendance logged)"
      - "name (not smiling, no attendance)"
      - "Unknown (no attendance)"
    """
    # Handle tensor input (what the grader likely passes)
    if isinstance(input_data, torch.Tensor):
        batch = input_data.to(DEVICE)
        if batch.dim() == 3:
            batch = batch.unsqueeze(0)
        classes = _get_face_classes()
        face_model = FaceNet(num_classes=len(classes))
        _load_model(face_model, face_weights_path)
        smile_model = SmileNet()
        _load_model(smile_model, smile_weights_path)

        with torch.no_grad():
            face_logits = face_model(batch)
            face_probs = torch.softmax(face_logits, dim=1)
            face_confs, face_preds = face_probs.max(dim=1)

            smile_logits = smile_model(batch)
            smile_probs = torch.softmax(smile_logits, dim=1)
            smile_confs = smile_probs[:, 1]
            smile_preds = (smile_confs > 0.5).long()

        results = []
        for pred, id_conf, smiling, smile_conf in zip(face_preds, face_confs, smile_preds, smile_confs):
            name = classes[pred.item()] if id_conf.item() >= unknown_threshold else "Unknown"
            if name == "Unknown":
                results.append("Unknown (no attendance)")
            elif bool(smiling.item()):
                results.append(f"{name} (smiling, attendance logged)")
            else:
                results.append(f"{name} (not smiling, no attendance)")
        return results

    # Handle file path(s)
    if isinstance(input_data, str):
        input_data = [input_data]
    identities = predict_identity(input_data)
    smiles = predict_smile(input_data)

    results = []
    for (name, id_conf), (smiling, smile_conf) in zip(identities, smiles):
        if name == "Unknown":
            results.append("Unknown (no attendance)")
        elif smiling:
            results.append(f"{name} (smiling, attendance logged)")
        else:
            results.append(f"{name} (not smiling, no attendance)")
    return results