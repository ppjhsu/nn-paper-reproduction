import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="runs/lightweight_cifar10_param_matched/log.csv")
    parser.add_argument("--out", type=str, default="runs/lightweight_cifar10_param_matched/training_curve.png")
    args = parser.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_path)

    plt.figure()
    plt.plot(df["epoch"], df["train_acc"] * 100, label="Train Acc")
    plt.plot(df["epoch"], df["test_acc"] * 100, label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Lightweight CIFAR-10 Classifier Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print("Saved:", out_path)

    loss_out = out_path.parent / "loss_curve.png"
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Lightweight CIFAR-10 Classifier Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_out, dpi=200, bbox_inches="tight")
    print("Saved:", loss_out)


if __name__ == "__main__":
    main()
