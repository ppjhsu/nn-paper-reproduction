import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model_param_matched import LightweightCIFAR10Classifier, count_parameters


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(batch_size: int, num_workers: int):
    """
    CIFAR-10:
        train: 50,000 images
        test : 10,000 images
        image size: 3 x 32 x 32

    論文沒有詳細列出 data augmentation。
    這裡使用 CIFAR-10 常見設定：
        RandomCrop(32, padding=4)
        RandomHorizontalFlip()
        Normalize()
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )

    test_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()

    # Paper setting:
    # epoch = 120, batch size = 64, optimizer = Adam
    # k = 89 is used here to match the paper reported parameter count (~0.26M).
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=89)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="runs/lightweight_cifar10_param_matched")

    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_loader, test_loader = get_loaders(args.batch_size, args.num_workers)

    model = LightweightCIFAR10Classifier(num_classes=10, k=args.k).to(device)
    total_params, trainable_params = count_parameters(model)

    print(f"Model k: {args.k}")
    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()

    # Paper uses Adam optimizer.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 論文未提供 learning rate scheduler。
    # 這裡加上 CosineAnnealingLR，通常能讓 CIFAR-10 訓練更穩。
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_path = out_dir / "log.csv"
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    best_acc = 0.0

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "lr",
            "train_loss",
            "train_acc",
            "test_loss",
            "test_acc",
            "best_acc",
            "params",
        ])

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            scheduler.step()

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "best_acc": best_acc,
                    "args": vars(args),
                    "params": total_params,
                }, best_path)

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_acc": best_acc,
                "args": vars(args),
                "params": total_params,
            }, last_path)

            lr = optimizer.param_groups[0]["lr"]

            writer.writerow([
                epoch,
                lr,
                train_loss,
                train_acc,
                test_loss,
                test_acc,
                best_acc,
                total_params,
            ])
            f.flush()

            print(
                f"Epoch [{epoch:03d}/{args.epochs}] "
                f"LR: {lr:.6f} "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_acc * 100:.2f}% "
                f"Test Loss: {test_loss:.4f} "
                f"Test Acc: {test_acc * 100:.2f}% "
                f"Best: {best_acc * 100:.2f}%"
            )

    print("Training finished.")
    print("Best accuracy:", best_acc)
    print("Saved best model to:", best_path)
    print("Saved log to:", log_path)


if __name__ == "__main__":
    main()
