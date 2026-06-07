import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s


def build_model(num_classes=10):
    model = efficientnet_v2_s(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./runs/effnetv2_cifar10/best.pt')
parser.add_argument('--img-size', type=int, default=160)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--repeat', type=int, default=50)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model().to(device)
ckpt = torch.load(args.ckpt, map_location=device)
model.load_state_dict(ckpt['model'])
model.eval()

x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
with torch.no_grad():
    for _ in range(10):
        _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.repeat):
        _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t = (time.time() - t0) / args.repeat

print(f'Device: {device}')
print(f'Average inference time per {args.batch_size} images: {t*1000:.2f} ms')
print(f'Average inference time per image: {t/args.batch_size*1000:.4f} ms')
