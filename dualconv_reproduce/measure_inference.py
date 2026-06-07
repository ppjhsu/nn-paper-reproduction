import argparse
import time
import torch
from models import build_model, count_parameters


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='./runs/dualconv/best.pt')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--model', default='vgg16')
    parser.add_argument('--kernel', default='dualconv', choices=['standard', 'dualconv', 'groupconv'])
    parser.add_argument('--g', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--repeat', type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10 if args.dataset == 'cifar10' else 100
    model = build_model(args.model, num_classes=num_classes, kernel=args.kernel, g=args.g).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    x = torch.randn(args.batch_size, 3, 32, 32, device=device)
    for _ in range(20):
        _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(args.repeat):
        _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    images = args.batch_size * args.repeat
    print(f'Params: {count_parameters(model):,}')
    print(f'Total images: {images}')
    print(f'Average inference time: {elapsed / images * 1000:.4f} ms/image')


if __name__ == '__main__':
    main()
