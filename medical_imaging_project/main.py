import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class MedicalImageDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_paths = sorted(self.image_dir.glob("*.png")) + sorted(self.image_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = 0 if "normal" in image_path.name.lower() else 1
        return image, torch.tensor(label, dtype=torch.long)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_dataloader(data_dir: str, batch_size: int = 4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = MedicalImageDataset(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_once(data_dir: str, epochs: int = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_dataloader(data_dir)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"epoch {epoch + 1}/{epochs} loss={total_loss / max(1, len(loader)):.4f}")

    return model


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print("请将医学影像图片放到 sample_images 目录中，文件名需包含 normal 或 abnormal。")
    else:
        train_once(data_dir, epochs=1)
