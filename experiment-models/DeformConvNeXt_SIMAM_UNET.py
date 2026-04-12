import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tvops


# ------------------------------------------------------------------
# DCNv2: Deformable Conv v2 with learned offsets + modulation mask
# Uses torchvision.ops.deform_conv2d (no extra deps required).
# groups is inferred from weight shape: groups = in_ch / weight.size(1)
# Offsets are spatially shared across groups (standard practice).
# ------------------------------------------------------------------
class DeformConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, groups: int = 1,
                 bias: bool = False):
        super().__init__()
        self.stride  = stride
        self.padding = padding
        k2 = kernel_size * kernel_size

        # Predict 2*k^2 offsets + k^2 modulation masks from input features
        self.offset_mask = nn.Conv2d(in_ch, 3 * k2, kernel_size=3, padding=1, bias=True)

        # weight shape: [out_ch, in_ch/groups, k, k]
        self.weight = nn.Parameter(
            torch.empty(out_ch, in_ch // groups, kernel_size, kernel_size)
        )
        self.bias_param = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self._k2 = k2

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Zero-init offsets/masks so the layer starts as a standard conv
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        om     = self.offset_mask(x)
        offset = om[:, :2 * self._k2]
        mask   = torch.sigmoid(om[:, 2 * self._k2:])
        return tvops.deform_conv2d(
            x, offset, self.weight, self.bias_param,
            stride=self.stride, padding=self.padding, mask=mask,
        )


# ------------------------------------------------------------------
# ConvNeXt block with deformable depthwise spatial mixing (DCNv2)
# DW-DeformConv(7x7) → LayerNorm → Linear expand → GELU → Linear project → γ·Δ + residual
# ------------------------------------------------------------------
class DeformConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7,
                 expand_ratio: int = 4, layer_scale_init: float = 1e-6):
        super().__init__()
        hidden = channels * expand_ratio

        # Depthwise deformable conv: groups=channels, weight [C, 1, k, k]
        self.dw_deform = DeformConv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm    = nn.LayerNorm(channels, eps=1e-6)
        self.pwconv1 = nn.Linear(channels, hidden)
        self.act     = nn.GELU()
        self.pwconv2 = nn.Linear(hidden, channels)
        self.gamma   = nn.Parameter(
            layer_scale_init * torch.ones(channels)
        ) if layer_scale_init > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw_deform(x)
        x = x.permute(0, 2, 3, 1)          # [B, H, W, C] for LN + Linear
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)          # [B, C, H, W]
        return residual + x


# ------------------------------------------------------------------
# SIMAM: Simple, Parameter-Free Attention (SimAM)
# ------------------------------------------------------------------
class SIMAM(nn.Module):
    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        n = h * w - 1
        mu = x.mean(dim=[2, 3], keepdim=True)
        x_c = x - mu
        var = (x_c ** 2).sum(dim=[2, 3], keepdim=True) / max(n, 1)
        e   = x_c ** 2 / (4 * (var + self.e_lambda)) + 0.5
        return x * torch.sigmoid(e)


# ------------------------------------------------------------------
# Standard DoubleConv  (inc, down1, down2, all decoder blocks)
# ------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ------------------------------------------------------------------
# DeformDoubleConv: 1st conv standard, 2nd conv DCNv2
# Used in deep encoder stages (down3 @ 32x32, down4 @ 16x16)
# ------------------------------------------------------------------
class DeformDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.dcn  = DeformConv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2  = nn.BatchNorm2d(out_ch)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return self.act2(self.bn2(self.dcn(x)))


# ------------------------------------------------------------------
# Encoder blocks
# ------------------------------------------------------------------
class Down(nn.Module):
    """MaxPool + DoubleConv (shallow encoder stages)"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class DeformDown(nn.Module):
    """MaxPool + DeformDoubleConv (deep encoder stages: down3, down4)"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DeformDoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


# ------------------------------------------------------------------
# Decoder block — bilinear upsample, SIMAM on skip, DoubleConv
# ------------------------------------------------------------------
class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int,
                 out_channels: int, bilinear: bool = True):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        self.attn = SIMAM()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy != 0 or dx != 0:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        skip = self.attn(skip)
        return self.conv(torch.cat([skip, x], dim=1))


# ------------------------------------------------------------------
# UNetDeformConvNeXtSimAM
#
# Encoder:
#   inc   (3→C)     256×256  — standard DoubleConv
#   down1 (C→2C)   128×128  — standard DoubleConv
#   down2 (2C→4C)  64×64    — standard DoubleConv
#   down3 (4C→8C)  32×32    — DeformDoubleConv  (DCNv2 on 2nd conv)
#   down4 (8C→16C) 16×16    — DeformDoubleConv  (DCNv2 on 2nd conv)
#
# Bottleneck:
#   DeformConvNeXtBlock (16C) — deformable DWConv(7×7) + MLP
#
# Decoder:
#   up1..up4 — bilinear upsample + SIMAM(skip) + DoubleConv
# ------------------------------------------------------------------
class UNetDeformConvNeXtSimAM(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_c: int = 64,
        bilinear: bool = True,
    ):
        super().__init__()

        # Shallow encoder — standard convs (high resolution, expensive for DCN)
        self.inc   = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c,       base_c * 2)
        self.down2 = Down(base_c * 2,   base_c * 4)

        # Deep encoder — DCNv2 on 2nd conv (32×32 and 16×16, manageable cost)
        self.down3 = DeformDown(base_c * 4,  base_c * 8)
        self.down4 = DeformDown(base_c * 8,  base_c * 16)

        # Bottleneck: deformable depthwise ConvNeXt block
        self.bottleneck = DeformConvNeXtBlock(
            channels=base_c * 16,
            kernel_size=7,
            expand_ratio=4,
            layer_scale_init=1e-6,
        )

        # Decoder with SIMAM on skip connections
        self.up1 = Up(base_c * 16, base_c * 8,  base_c * 8,  bilinear)
        self.up2 = Up(base_c * 8,  base_c * 4,  base_c * 4,  bilinear)
        self.up3 = Up(base_c * 4,  base_c * 2,  base_c * 2,  bilinear)
        self.up4 = Up(base_c * 2,  base_c,      base_c,      bilinear)

        self.outc = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)       # [B,   C, 256, 256]
        x2 = self.down1(x1)    # [B,  2C, 128, 128]
        x3 = self.down2(x2)    # [B,  4C,  64,  64]
        x4 = self.down3(x3)    # [B,  8C,  32,  32]  DCNv2
        x5 = self.down4(x4)    # [B, 16C,  16,  16]  DCNv2

        # Bottleneck
        x5 = self.bottleneck(x5)   # deformable DWConv + MLP

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)

        return self.outc(x)


if __name__ == "__main__":
    model = UNetDeformConvNeXtSimAM(in_channels=3, num_classes=1, base_c=64)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print("Input :", x.shape)
    print("Output:", y.shape)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
