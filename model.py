import torch
import torch.nn as nn
from torchvision import models
from config import DEVICE, smile_num_classes


class FaceNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class SmileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, smile_num_classes)

    def forward(self, x):
        return self.backbone(x)


class SmileInModel(nn.Module):
    def __init__(self, num_face_classes):
        super().__init__()
        self.face_net = FaceNet(num_face_classes)
        self.smile_net = SmileNet()

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