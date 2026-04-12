import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# ConvNeXt Block
# Reference: https://arxiv.org/abs/2201.03545
# ------------------------------------------------------------------
class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block:
      DWConv (7x7) -> LayerNorm -> Linear (expand) -> GELU -> Linear (project) -> layer scale -> residual
    """
    def __init__(self, channels: int, kernel_size: int = 7,
                 expand_ratio: int = 4, layer_scale_init: float = 1e-6):
        super().__init__()
        hidden = channels * expand_ratio
        self.dwconv  = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                                 padding=kernel_size // 2, groups=channels)
        self.norm    = nn.LayerNorm(channels, eps=1e-6)
        self.pwconv1 = nn.Linear(channels, hidden)
        self.act     = nn.GELU()
        self.pwconv2 = nn.Linear(hidden, channels)
        self.gamma   = nn.Parameter(
            layer_scale_init * torch.ones(channels)
        ) if layer_scale_init > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)          # [B, H, W, C] for LayerNorm + Linear
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
    """
    SimAM: A Simple, Parameter-free Attention Module
    Applied element-wise using an energy function.
    Paper: https://arxiv.org/abs/2103.06215
    """
    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        b, c, h, w = x.size()
        n = h * w - 1

        # Center w.r.t channel-wise spatial mean
        mu = x.mean(dim=[2, 3], keepdim=True)
        x_centered = x - mu

        var = (x_centered ** 2).sum(dim=[2, 3], keepdim=True) / max(n, 1)
        e = x_centered ** 2 / (4 * (var + self.e_lambda)) + 0.5

        attn = torch.sigmoid(e)
        return x * attn


# ------------------------------------------------------------------
# Basic UNet building blocks
# ------------------------------------------------------------------
class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """
    Upscaling then double conv.
    Uses bilinear upsampling by default.
    SIMAM is applied to the skip connection before concatenation.
    """
    def __init__(
        self,
        in_channels: int,     # channels coming from previous (lower) layer
        skip_channels: int,   # channels from encoder skip
        out_channels: int,
        bilinear: bool = True
    ):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            # If you prefer transposed conv, uncomment and adjust channels
            # self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            # For simplicity, keep bilinear for now:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # After upsample: concat([skip, up(x)]) -> DoubleConv
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        self.attn = SIMAM()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x: from decoder lower level
        # skip: from encoder (same spatial level)
        x = self.up(x)

        # Handle size mismatch if input is not divisible by 2 multiple times
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2,
                 diff_y // 2, diff_y - diff_y // 2]
            )

        # Apply SIMAM on skip connection
        skip = self.attn(skip)

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


# ------------------------------------------------------------------
# UNet with ConvNeXt bottleneck + SIMAM on skip connections
# ------------------------------------------------------------------
class UNetConvNeXtSimAM(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_c: int = 64,
        bilinear: bool = True,
    ):
        super().__init__()

        # Encoder
        self.inc   = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c,       base_c * 2)
        self.down2 = Down(base_c * 2,   base_c * 4)
        self.down3 = Down(base_c * 4,   base_c * 8)
        self.down4 = Down(base_c * 8,   base_c * 16)

        # Bottleneck: ConvNeXt block (depthwise 7x7 + channel MLP + layer scale)
        self.bottleneck = ConvNeXtBlock(
            channels=base_c * 16,
            kernel_size=7,
            expand_ratio=4,
            layer_scale_init=1e-6,
        )

        # Decoder with SIMAM on skips
        self.up1 = Up(
            in_channels=base_c * 16,
            skip_channels=base_c * 8,
            out_channels=base_c * 8,
            bilinear=bilinear,
        )
        self.up2 = Up(
            in_channels=base_c * 8,
            skip_channels=base_c * 4,
            out_channels=base_c * 4,
            bilinear=bilinear,
        )
        self.up3 = Up(
            in_channels=base_c * 4,
            skip_channels=base_c * 2,
            out_channels=base_c * 2,
            bilinear=bilinear,
        )
        self.up4 = Up(
            in_channels=base_c * 2,
            skip_channels=base_c,
            out_channels=base_c,
            bilinear=bilinear,
        )

        self.outc = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)       # [B, base_c, H,   W  ]
        x2 = self.down1(x1)    # [B, 2C,    H/2, W/2]
        x3 = self.down2(x2)    # [B, 4C,    H/4, W/4]
        x4 = self.down3(x3)    # [B, 8C,    H/8, W/8]
        x5 = self.down4(x4)    # [B,16C,   H/16, W/16]

        # Bottleneck with InvHead (Involution inside)
        x5 = self.bottleneck(x5)

        # Decoder with SIMAM skip attention
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)

        logits = self.outc(x)
        return logits


# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy input
    x = torch.randn(1, 1, 256, 256)

    # Make sure your Involution + InvHead definitions are above this.
    model = UNetConvNeXtSimAM(in_channels=1, num_classes=1, base_c=32)
    y = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)



