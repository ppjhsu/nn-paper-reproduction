import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from model_param_matched import LightweightCIFAR10Classifier


class CIFAR10CDataset(Dataset):
    def __init__(self, data_path, label_path, severity=1):
        """
        CIFAR-10-C 每個 corruption 檔案通常有 50,000 張圖片。
        分成 5 個 severity level，每個 level 10,000 張。
        severity = 1 使用第 0~9999 張
        severity = 2 使用第 10000~19999 張
        ...
        """
        self.data = np.load(data_path)
        self.labels = np.load(label_path)

        assert severity in [1, 2, 3, 4, 5], "severity 必須是 1 到 5"

        start = (severity - 1) * 10000
        end = severity * 10000

        self.data = self.data[start:end]
        self.labels = self.labels[start:end]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            ),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = int(self.labels[idx])

        image = self.transform(image)

        return image, label


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    return acc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, default="runs/lightweight_cifar10_param_matched/best.pt")
    parser.add_argument("--cifar10c-dir", type=str, default="data/CIFAR-10-C")
    parser.add_argument("--corruption", type=str, default="brightness")
    parser.add_argument("--severity", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--k", type=int, default=89)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    cifar10c_dir = Path(args.cifar10c_dir)
    data_path = cifar10c_dir / f"{args.corruption}.npy"
    label_path = cifar10c_dir / "labels.npy"

    if not data_path.exists():
        raise FileNotFoundError(f"找不到 corruption 檔案：{data_path}")

    if not label_path.exists():
        raise FileNotFoundError(f"找不到 labels.npy：{label_path}")

    checkpoint = torch.load(args.ckpt, map_location=device)

    if "args" in checkpoint and "k" in checkpoint["args"]:
        k = checkpoint["args"]["k"]
    else:
        k = args.k

    model = LightweightCIFAR10Classifier(num_classes=10, k=k).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("Loaded checkpoint:", args.ckpt)
    print("Best CIFAR-10 accuracy:", checkpoint.get("best_acc", None))

    dataset = CIFAR10CDataset(
        data_path=data_path,
        label_path=label_path,
        severity=args.severity,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    acc = evaluate(model, loader, device)

    print("=" * 50)
    print("CIFAR-10-C Test Result")
    print("Corruption:", args.corruption)
    print("Severity:", args.severity)
    print(f"Accuracy: {acc * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()