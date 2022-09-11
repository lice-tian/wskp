import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from typing import List
from repvgg_utils import get_fused_conv_state_dict_from_block


class RepVGGFastBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=5, stride=stride, padding=2
        )
        self.relu = nn.ReLU(inplace=True)


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.branch5 = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=5,
            padding=2,
            bias=False,
            stride=stride,
            activation_layer=None
        )

        self.branch3 = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            stride=stride,
            activation_layer=None,
        )

        self.branch1 = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            activation_layer=None
        )

        self.identity = (
            nn.BatchNorm2d(out_channels) if in_channels == out_channels else None
        )

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = x
        x = self.branch1(x)
        x += self.branch3(res)
        x += self.branch5(res)
        if self.identity:
            x += self.identity(res)
        x = self.relu(x)
        return x

    def to_fast(self) -> RepVGGFastBlock:
        fused_conv_state_dict = get_fused_conv_state_dict_from_block(self)
        fast_block = RepVGGFastBlock(
            self.branch5[0].in_channels,
            self.branch5[0].out_channels,
            stride=self.branch5[0].stride
        )
        fast_block.conv.load_state_dict(fused_conv_state_dict)

        return fast_block


class RepVGG(nn.Sequential):
    def __init__(self, channels: List[int]):
        super().__init__()
        in_out_channels = zip(channels, channels[1:])

        self.blocks = nn.Sequential(
            *[
                RepVGGBlock(in_channels, out_channels, stride=1)
                for in_channels, out_channels in in_out_channels
            ],
        )

        # omit classification head for simplicity

    def switch_to_fast(self):
        for i, block in enumerate(self.blocks):
            self.blocks[i] = block.to_fast()
        return self


if __name__ == '__main__':
    with torch.no_grad():
        repvgg = RepVGG([1, 14, 12])
        repvgg.switch_to_fast()
        repvgg = repvgg.eval()
        repvgg.to('cuda')
        t = torch.randn((3, 1, 10, 10))
        t = t.to('cuda')
        t = repvgg(t)
        print(t.shape)
