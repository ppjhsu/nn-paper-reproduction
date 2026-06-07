import argparse
import time
from pathlib import Path

import torch

from model_param_matched import LightweightCIFAR10Classifier, count_parameters


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/lightweight_cifar10_param_matched/best.pt")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--k", type=int, default=89)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LightweightCIFAR10Classifier(num_classes=10, k=args.k).to(device)

    ckpt_path = Path(args.ckpt)
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print("Loaded checkpoint:", ckpt_path)
        print("Best accuracy in checkpoint:", checkpoint.get("best_acc", None))
    else:
        print("Checkpoint not found. Measuring random initialized model.")

    model.eval()

    total_params, _ = count_parameters(model)
    print(f"Parameters: {total_params:,}")

    x = torch.randn(args.batch_size, 3, 32, 32).to(device)

    # warm up
    for _ in range(10):
        _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(args.repeat):
        _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    total_time = end - start
    avg_batch_time = total_time / args.repeat
    avg_image_time = avg_batch_time / args.batch_size

    print("Device:", device)
    print(f"Batch size: {args.batch_size}")
    print(f"Repeat: {args.repeat}")
    print(f"Average time per batch: {avg_batch_time * 1000:.4f} ms")
    print(f"Average time per image: {avg_image_time * 1000:.6f} ms")


if __name__ == "__main__":
    main()
