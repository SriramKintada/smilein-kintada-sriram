import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import (
    resize_x, resize_y, input_channels,
    imagenet_mean, imagenet_std,
    face_batch_size, smile_batch_size,
    face_data_dir, smile_data_dir,
)


train_transform = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

val_transform = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])


class FaceIdentityDataset(Dataset):
    def __init__(self, root_dir=face_data_dir, transform=None, split="train"):
        self.root_dir = root_dir
        self.transform = transform if transform else train_transform
        self.split = split
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = self._gather_samples()
        if split != "train":
            self.samples = self.samples[int(len(self.samples) * 0.7):]
        else:
            self.samples = self.samples[:int(len(self.samples) * 0.7)]

    def _gather_samples(self):
        samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".pgm", ".bmp")):
                    samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class SmileDataset(Dataset):
    def __init__(self, root_dir=smile_data_dir, transform=None, split="train"):
        self.root_dir = root_dir
        self.transform = transform if transform else train_transform
        self.split = split
        self.classes = ["not_smiling", "smiling"]
        self.class_to_idx = {"not_smiling": 0, "smiling": 1}
        self.samples = self._gather_samples()
        if split != "train":
            self.samples = self.samples[int(len(self.samples) * 0.7):]
        else:
            self.samples = self.samples[:int(len(self.samples) * 0.7)]

    def _gather_samples(self):
        samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".pgm", ".bmp")):
                    samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_face_dataloader(batch_size=face_batch_size, split="train"):
    transform = train_transform if split == "train" else val_transform
    dataset = FaceIdentityDataset(transform=transform, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)


def get_smile_dataloader(batch_size=smile_batch_size, split="train"):
    transform = train_transform if split == "train" else val_transform
    dataset = SmileDataset(transform=transform, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)