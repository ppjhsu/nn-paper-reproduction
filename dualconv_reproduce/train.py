import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import build_model, count_parameters


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def accuracy(output, target):
    pred = output.argmax(dim=1)
    return (pred == target).float().mean().item() * 100.0


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == targets).sum().item()
        total += images.size(0)
    return total_loss / total, 100.0 * total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == targets).sum().item()
        total += images.size(0)
    return total_loss / total, 100.0 * total_correct / total


def main():
    parser = argparse.ArgumentParser(description='Reproduce DualConv on CIFAR-10/CIFAR-100')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--model', default='vgg16')
    parser.add_argument('--kernel', default='dualconv', choices=['standard', 'dualconv', 'groupconv'])
    parser.add_argument('--g', type=int, default=4, help='group number G for DualConv/GroupConv')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--out-dir', default='./runs/dualconv')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    if args.dataset == 'cifar10':
        Dataset = datasets.CIFAR10
        num_classes = 10
    else:
        Dataset = datasets.CIFAR100
        num_classes = 100

    train_set = Dataset(root=args.data_dir, train=True, download=True, transform=train_tf)
    test_set = Dataset(root=args.data_dir, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model, num_classes=num_classes, kernel=args.kernel, g=args.g).to(device)
    print(f'Model: {args.model}_{args.kernel}_G{args.g} | Params: {count_parameters(model):,} | Device: {device}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # Paper for CIFAR-10: initial lr=0.1 and multiply by 0.1 every 50 epochs.
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)

    log_path = out_dir / 'log.csv'
    best_acc = 0.0
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'lr', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'time_sec'])
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            scheduler.step()
            elapsed = time.time() - start
            lr_now = optimizer.param_groups[0]['lr']
            writer.writerow([epoch, lr_now, train_loss, train_acc, test_loss, test_acc, elapsed])
            f.flush()
            print(f'Epoch {epoch:03d}/{args.epochs} | train {train_acc:.2f}% | test {test_acc:.2f}% | loss {test_loss:.4f} | {elapsed:.1f}s')
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': vars(args),
            }
            torch.save(ckpt, out_dir / 'last.pt')
            if test_acc > best_acc:
                best_acc = test_acc
                ckpt['best_acc'] = best_acc
                torch.save(ckpt, out_dir / 'best.pt')
    print(f'Best test acc: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
