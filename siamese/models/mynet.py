import torch
import torch.nn as nn
from torchsummary import summary


def set_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


# * Basic Layers
class Conv(nn.Module):
    """
    Basic conv layer (conv + bn + lrelu).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        IN=False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
        )
        self.bn = nn.GroupNorm(in_channels // 2, out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    """
    3x3 Depthwise Separable Convolution layer (depthwise + pointwise + bn + lrelu).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        padding=1,
        IN=False,
    ):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False,
        )
        self.bn = nn.GroupNorm(in_channels // 2, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """
    1x1 convolution + bn + relu.
    """

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """
    1x1 convolution + bn without activation layer.
    """

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BottleneckBlock(nn.Module):
    """
    Combined conv block wit Conv and DepthwiseSeparableConv
    """

    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        mid_channels = in_channels // reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = DepthwiseSeparableConv(mid_channels, mid_channels)
        self.conv2b = DepthwiseSeparableConv(
            mid_channels, mid_channels, IN=True
        )
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        # if in_channels != out_channels:
        #     self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.bn = nn.GroupNorm(in_channels // 4, out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block(nn.Module):
    """
    Custom block implementation
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        reduction,
        padding=0,
    ):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size, padding)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.bottleneck = BottleneckBlock(
            out_channels, out_channels, reduction=reduction
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bottleneck(x)
        return x


class MyNet(nn.Module):
    """
    Custom Network.
    """

    def __init__(
        self,
        in_channels=3,
        conv_channels=[8, 16, 32],
        num_classes=1,
        feature_dims=512,
        loss="softmax",
    ):
        super().__init__()
        self.loss = loss
        self.feature_dims = feature_dims
        self.block1 = Block(in_channels, conv_channels[0], 5, 4, padding=1)
        self.block2 = Block(conv_channels[0], conv_channels[1], 3, 4, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.block3 = Block(conv_channels[1], conv_channels[2], 3, 4, padding=1)
        self.dropout2 = nn.Dropout(p=0.3)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._make_fc_layer(
            conv_channels[2], self.feature_dims
        )
        # self.classifier = nn.Linear(self.feature_dims, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.dropout1(x)
        x = self.block3(x)
        x = self.dropout2(x)
        x = self.global_avgpool(x)
        x = self.fc(x)
        # x = self.classifier(x)
        return x

    # * Utility fucntions
    def _make_fc_layer(self, input_dims, fc_dims):
        layers = []
        layers.append(nn.Flatten())
        layers.append(nn.Linear(input_dims, fc_dims))
        layers.append(nn.LeakyReLU(inplace=True))

        return nn.Sequential(*layers)


if __name__ == "__main__":
    model = MyNet(in_channels=3, conv_channels=[8, 16, 32], num_classes=1)
    summary(model, (3, 256, 128))
