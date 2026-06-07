import torch
import torch.nn as nn
from dualconv import DualConv


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class VGG16CIFAR(nn.Module):
    """VGG-16 for CIFAR.

    The paper replaces the last 12 3x3 convolution layers of VGG-16 with
    DualConv on CIFAR-10. VGG-16 has 13 conv layers, so only the first conv
    remains a normal convolution.
    """

    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

    def __init__(self, num_classes: int = 10, kernel: str = 'dualconv', g: int = 4, batch_norm: bool = True):
        super().__init__()
        self.features = self._make_layers(kernel=kernel, g=g, batch_norm=batch_norm)
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_layers(self, kernel: str, g: int, batch_norm: bool):
        layers = []
        in_channels = 3
        conv_index = 0
        for v in self.cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv_index += 1
                if kernel == 'dualconv' and conv_index >= 2:
                    conv = DualConv(in_channels, v, stride=1, g=g, bias=False)
                elif kernel == 'groupconv' and conv_index >= 2:
                    valid_g = g if in_channels % g == 0 and v % g == 0 else 1
                    conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, groups=valid_g, bias=False)
                else:
                    conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv, nn.ReLU(inplace=True)]
                in_channels = v
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, DualConv)):
                # DualConv contains its own Conv2d modules, initialized below.
                continue
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def build_model(model_name: str = 'vgg16', num_classes: int = 10, kernel: str = 'dualconv', g: int = 4):
    model_name = model_name.lower()
    if model_name in ['vgg', 'vgg16']:
        return VGG16CIFAR(num_classes=num_classes, kernel=kernel, g=g)
    raise ValueError(f'Unsupported model: {model_name}')
