import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='./runs/dualconv/log.csv')
    parser.add_argument('--out', default='./runs/dualconv/training_curve.png')
    args = parser.parse_args()
    df = pd.read_csv(args.log)
    plt.figure()
    plt.plot(df['epoch'], df['train_acc'], label='Train Acc')
    plt.plot(df['epoch'], df['test_acc'], label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
