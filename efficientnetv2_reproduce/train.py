import argparse
import csv
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_stage_values(epoch, epochs, min_v, max_v):
    if epochs <= 1:
        return max_v
    t = epoch / (epochs - 1)
    return min_v + (max_v - min_v) * t


def make_train_transform(img_size: int, randaug_mag: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomCrop(img_size, padding=max(4, img_size // 16)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=int(randaug_mag)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


def make_test_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


def mixup_data(x, y, alpha: float):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    if y_b is None:
        return criterion(pred, y_a)
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def set_classifier_dropout(model: nn.Module, p: float):
    # torchvision EfficientNet classifier: Dropout + Linear
    for m in model.classifier.modules():
        if isinstance(m, nn.Dropout):
            m.p = float(p)


def build_model(num_classes: int, pretrained: bool):
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_v2_s(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="EfficientNetV2-S CIFAR-10 reproduction with progressive learning")
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--out-dir', type=str, default='./runs/effnetv2_cifar10')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained weights if torchvision can download/cache them')
    parser.add_argument('--min-size', type=int, default=96)
    parser.add_argument('--max-size', type=int, default=160)
    parser.add_argument('--min-randaug', type=int, default=5)
    parser.add_argument('--max-randaug', type=int, default=15)
    parser.add_argument('--min-dropout', type=float, default=0.05)
    parser.add_argument('--max-dropout', type=float, default=0.20)
    parser.add_argument('--min-mixup', type=float, default=0.0)
    parser.add_argument('--max-mixup', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(num_classes=10, pretrained=args.pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    final_test_transform = make_test_transform(args.max_size)
    test_set = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=final_test_transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    log_path = out_dir / 'log.csv'
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'img_size', 'randaug_mag', 'dropout', 'mixup_alpha', 'lr',
                         'train_loss', 'train_acc', 'val_loss', 'val_acc', 'seconds'])

    best_acc = 0.0
    for epoch in range(args.epochs):
        img_size = int(round(get_stage_values(epoch, args.epochs, args.min_size, args.max_size)))
        randaug_mag = int(round(get_stage_values(epoch, args.epochs, args.min_randaug, args.max_randaug)))
        dropout = float(get_stage_values(epoch, args.epochs, args.min_dropout, args.max_dropout))
        mixup_alpha = float(get_stage_values(epoch, args.epochs, args.min_mixup, args.max_mixup))
        set_classifier_dropout(model, dropout)

        train_transform = make_train_transform(img_size, randaug_mag)
        train_set = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)

        model.train()
        t0 = time.time()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x_mix, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)

            optimizer.zero_grad(set_to_none=True)
            out = model(x_mix)
            loss = mixup_loss(criterion, out, y_a, y_b, lam)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            # For logging only; mixup accuracy is approximate, use original labels.
            correct += (pred == y).sum().item()
            total += x.size(0)

        scheduler.step()
        train_loss = loss_sum / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, test_loader, device)
        seconds = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1:03d}/{args.epochs} | size={img_size} aug={randaug_mag} "
              f"drop={dropout:.3f} mixup={mixup_alpha:.3f} | "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} time={seconds:.1f}s")

        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, img_size, randaug_mag, dropout, mixup_alpha, lr_now,
                             train_loss, train_acc, val_loss, val_acc, seconds])

        torch.save({'model': model.state_dict(), 'args': vars(args), 'epoch': epoch + 1,
                    'val_acc': val_acc}, out_dir / 'last.pt')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model': model.state_dict(), 'args': vars(args), 'epoch': epoch + 1,
                        'val_acc': val_acc}, out_dir / 'best.pt')

    print(f"Best val_acc = {best_acc:.4f}")
    print(f"Saved to: {out_dir}")


if __name__ == '__main__':
    main()
