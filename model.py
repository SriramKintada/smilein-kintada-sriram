import os
import torch
import torch.nn as nn
from torchvision import models
from config import DEVICE, smile_num_classes, face_data_dir, face_weights_path, smile_weights_path


def _get_num_face_classes():
    if os.path.isdir(face_data_dir):
        return len([d for d in os.listdir(face_data_dir)
                     if os.path.isdir(os.path.join(face_data_dir, d))])
    return 6


class FaceNet(nn.Module):
    def __init__(self, num_classes=_get_num_face_classes()):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        # Freeze early layers so the model learns facial features, not photo style
        for param in self.backbone.features[:14].parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


class SmileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, smile_num_classes)
        # Freeze early layers for smile detection too
        for param in self.backbone.features[:10].parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


class SmileInModel(nn.Module):
    def __init__(self, num_face_classes=_get_num_face_classes()):
        super().__init__()
        self.face_net = FaceNet(num_face_classes)
        self.smile_net = SmileNet()
        if os.path.exists(face_weights_path):
            self.face_net.load_state_dict(torch.load(face_weights_path, map_location=DEVICE, weights_only=True))
        if os.path.exists(smile_weights_path):
            self.smile_net.load_state_dict(torch.load(smile_weights_path, map_location=DEVICE, weights_only=True))

    def forward(self, x):
        face_logits = self.face_net(x)
        smile_logits = self.smile_net(x)
        return face_logits, smile_logits

    def predict_face(self, x):
        self.face_net.eval()
        with torch.no_grad():
            logits = self.face_net(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
        return pred, conf

    def predict_smile(self, x):
        self.smile_net.eval()
        with torch.no_grad():
            logits = self.smile_net(x)
            probs = torch.softmax(logits, dim=1)
            smile_conf = probs[:, 1]
            smile_pred = (smile_conf > 0.5).long()
        return smile_pred, smile_conf