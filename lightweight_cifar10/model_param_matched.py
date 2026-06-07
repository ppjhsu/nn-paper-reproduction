import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """
    Conv -> BatchNorm -> ReLU

    論文說明：
    All the depthwise and pointwise convolutional layers are followed by
    batch normalization and ReLU activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MainBlock(nn.Module):
    """
    Main_block from the paper.

    論文 Fig. 2：
    Previous Layer 被分成三條路徑：

    Path 1:
        PointWiseConv(k) -> DepthwiseConv -> PointWiseConv(k)

    Path 2:
        DepthwiseConv -> PointWiseConv(2*k)

    Path 3:
        Identity shortcut, directly goes to concatenation.

    Output:
        Concatenation(Path1, Path2, Path3)
    """
    def __init__(self, in_channels, k=64):
        super().__init__()
        self.path1 = nn.Sequential(
            ConvBNReLU(in_channels, k, kernel_size=1),
            ConvBNReLU(k, k, kernel_size=3, groups=k),
            ConvBNReLU(k, k, kernel_size=1),
        )

        self.path2 = nn.Sequential(
            ConvBNReLU(in_channels, in_channels, kernel_size=3, groups=in_channels),
            ConvBNReLU(in_channels, 2 * k, kernel_size=1),
        )

        self.out_channels = k + (2 * k) + in_channels

    def forward(self, x):
        y1 = self.path1(x)
        y2 = self.path2(x)
        y3 = x
        return torch.cat([y1, y2, y3], dim=1)


class TransitionBlock(nn.Module):
    """
    Transition_block from the paper.

    論文 Fig. 3：
        PointWiseConv(k) -> Average Pooling 2x2

    功能：
        1. 用 1x1 pointwise convolution 壓縮通道數
        2. 用 AvgPool2d(2) 做 downsampling
    """
    def __init__(self, in_channels, k=64):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, k, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.out_channels = k

    def forward(self, x):
        return self.block(x)


class LightweightCIFAR10Classifier(nn.Module):
    """
    Reproduction implementation of:
    "Lightweight image classifier for CIFAR-10"
    Akshay Kumar Sharma, Amrita Rana, Kyung Ki Kim, 2021.

    重要說明：
    論文沒有公開完整官方程式碼，只有架構圖與 pseudo-code。
    原文說 CIFAR-10 中 k=64，但依照公開圖與 pseudo-code 直接實作時，
    參數量約 0.14M，低於論文表一的 0.26M。
    因此本版本將 k 調整為 89，使模型總參數量接近論文 reported 0.26M。
    這是「參數量對齊版本」，不是官方完全一致版本。

    Architecture:
        Initial PointWiseConv
        [MainBlock + TransitionBlock] x 3
        MainBlock x 1
        PointWiseConv
        Global Average Pooling
        Dropout(0.5)
        Dense Softmax / Linear classifier
    """
    def __init__(self, num_classes=10, k=89):
        super().__init__()

        self.k = k  # k=89: parameter-matched setting, close to paper reported 0.26M params

        # Initial layer: pointwise conv + BN + ReLU
        self.initial = ConvBNReLU(3, k, kernel_size=1)

        # Repetition 1
        self.main1 = MainBlock(k, k=k)
        self.trans1 = TransitionBlock(self.main1.out_channels, k=k)

        # Repetition 2
        self.main2 = MainBlock(k, k=k)
        self.trans2 = TransitionBlock(self.main2.out_channels, k=k)

        # Repetition 3
        self.main3 = MainBlock(k, k=k)
        self.trans3 = TransitionBlock(self.main3.out_channels, k=k)

        # Repetition 4: only MainBlock, as described in the paper
        self.main4 = MainBlock(k, k=k)

        # After final repetition: pointwise conv -> GAP -> dropout -> dense
        self.final_pointwise = ConvBNReLU(self.main4.out_channels, k, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(k, num_classes)

    def forward(self, x):
        x = self.initial(x)

        x = self.main1(x)
        x = self.trans1(x)

        x = self.main2(x)
        x = self.trans2(x)

        x = self.main3(x)
        x = self.trans3(x)

        x = self.main4(x)

        x = self.final_pointwise(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = LightweightCIFAR10Classifier(num_classes=10, k=89)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)

    total, trainable = count_parameters(model)
    print(model)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print(f"Total parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")
