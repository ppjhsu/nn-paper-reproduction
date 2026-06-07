import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='./runs/effnetv2_cifar10/log.csv')
args = parser.parse_args()

log_path = Path(args.log)
df = pd.read_csv(log_path)
out_dir = log_path.parent

plt.figure()
plt.plot(df['epoch'], df['train_acc'], label='train_acc')
plt.plot(df['epoch'], df['val_acc'], label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(out_dir / 'accuracy_curve.png', dpi=200, bbox_inches='tight')

plt.figure()
plt.plot(df['epoch'], df['train_loss'], label='train_loss')
plt.plot(df['epoch'], df['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(out_dir / 'loss_curve.png', dpi=200, bbox_inches='tight')

print(f'Saved curves to {out_dir}')
