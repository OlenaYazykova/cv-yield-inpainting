import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_num_groups(num_channels: int, max_groups: int = 8) -> int:
    """
    Finds the number of groups for GroupNorm such that
    num_channels is divisible by num_groups.
    """
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


def _pad_to_match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Pads x to match the spatial size of ref.
    Required to prevent skip-connections from breaking on non-ideal spatial dimensions.
    """
    diff_y = ref.size(2) - x.size(2)
    diff_x = ref.size(3) - x.size(3)

    if diff_y == 0 and diff_x == 0:
        return x

    x = F.pad(
        x,
        [
            diff_x // 2,
            diff_x - diff_x // 2,
            diff_y // 2,
            diff_y - diff_y // 2,
        ],
    )
    return x


class DoubleConv(nn.Module):
    """
    Conv -> GroupNorm -> ReLU -> Conv -> GroupNorm -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int, max_groups: int = 8):
        super().__init__()

        groups = _get_num_groups(out_channels, max_groups)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = _pad_to_match(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class ShallowUNet(nn.Module):
    """
    Baseline:
    3 encoder-уровня + bottleneck
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 16,
    ):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DownBlock(base_channels, base_channels * 2)
        self.enc3 = DownBlock(base_channels * 2, base_channels * 4)

        self.bottleneck = DownBlock(base_channels * 4, base_channels * 8)

        self.up3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        out = self.out_conv(d1)
        return out


class DeepUNet(nn.Module):
    """
    Deeper model:
    4 encoder levels + bottleneck
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 32,
    ):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DownBlock(base_channels, base_channels * 2)
        self.enc3 = DownBlock(base_channels * 2, base_channels * 4)
        self.enc4 = DownBlock(base_channels * 4, base_channels * 8)

        self.bottleneck = DownBlock(base_channels * 8, base_channels * 16)

        self.up4 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        out = self.out_conv(d1)
        return out


def build_model(
    model_name: str,
    in_channels: int,
    out_channels: int = 1,
    base_channels: int | None = None,
) -> nn.Module:
    """
    Unified model factory.
    model_name:
        - "baseline" / "shallow" — ShallowUNet, 3 encoder levels
        - "deep" — DeepUNet, 4 encoder levels
    """
    model_name = model_name.lower()

    if model_name in ["baseline", "shallow"]:
        if base_channels is None:
            base_channels = 16
        return ShallowUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )

    if model_name == "deep":
        if base_channels is None:
            base_channels = 32
        return DeepUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )

    raise ValueError(f"Unknown model_name: {model_name}")
