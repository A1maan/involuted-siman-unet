import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Involution (group = input channels) ----------
# --- changes only: add 'groups' to Involution/InvHead/MobileNetV3_INV ---

class Involution(nn.Module):
    def __init__(self, channels, kernel_size=7, stride=1, reduction=4,
                 kernel_norm: str = "l2", softmax_temp: float = 1.0,
                 groups: int | None = None):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction = max(1, reduction)
        self.kernel_norm = kernel_norm
        self.softmax_temp = softmax_temp

        # groups: number of channel groups for dynamic kernels
        self.groups = channels if (groups is None) else int(groups)
        assert self.groups >= 1 and channels % self.groups == 0, \
            f"'groups' must divide channels: got C={channels}, groups={self.groups}"

        hidden = max(1, channels // self.reduction)

        # C -> hidden -> (k^2 * groups)
        self.reduce = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.bn     = nn.BatchNorm2d(hidden)
        self.act    = nn.ReLU(inplace=True)
        self.kproj  = nn.Conv2d(hidden, (kernel_size * kernel_size) * self.groups,
                                kernel_size=1, bias=True)

        self.pool_for_k = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()

    def _normalize_kernel(self, ker: torch.Tensor) -> torch.Tensor:
        if self.kernel_norm == "softmax":
            B, G, k, _, H, W = ker.shape
            ker = F.softmax(ker.view(B, G, k*k, H, W) / self.softmax_temp, dim=2)
            return ker.view(B, G, k, k, H, W)
        if self.kernel_norm == "l2":
            ker = ker - ker.mean(dim=(2, 3), keepdim=True)
            denom = ker.norm(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            ker = ker / denom
        return ker

    @torch.no_grad()
    def get_kernels(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k = self.kernel_size
        xk = self.pool_for_k(x)
        K  = self.kproj(self.act(self.bn(self.reduce(xk))))            # [B, k^2*G, H', W']
        if K.shape[-2:] != (H, W):
            K = F.interpolate(K, size=(H, W), mode="bilinear", align_corners=True)
        K = K.view(B, self.groups, k, k, H, W)                         # [B,G,k,k,H,W]
        return self._normalize_kernel(K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k, G = self.kernel_size, self.groups
        assert C % G == 0
        groupC = C // G

        K = self.get_kernels(x)                                        # [B,G,k,k,H,W]
        x_unfold = F.unfold(x, kernel_size=k, padding=k//2)            # [B,C*k*k,H*W]
        x_unfold = x_unfold.view(B, C, k, k, H, W).view(B, G, groupC, k, k, H, W)
        out = (x_unfold * K.unsqueeze(2)).sum(dim=(3,4)).view(B, C, H, W)
        return out


class InvHead(nn.Module):
    def __init__(self, channels, reduce=4, k=9, inv_reduction=4,
                 kernel_norm="l2", softmax_temp=1.0, inv_groups: int | None = None):
        super().__init__()
        hidden = max(8, channels // reduce)
        self.hidden = hidden

        self.reduce = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn1    = nn.BatchNorm2d(hidden)
        self.act    = nn.ReLU(inplace=True)

        # 'inv_groups' applies over HIDDEN channels
        if inv_groups is None:
            inv_groups = hidden  # depthwise by default
        else:
            inv_groups = int(inv_groups)
        assert hidden % inv_groups == 0, \
            f"'inv_groups' must divide hidden={hidden}; got {inv_groups}"

        self.inv = Involution(
            channels=hidden, kernel_size=k, stride=1,
            reduction=inv_reduction, kernel_norm=kernel_norm,
            softmax_temp=softmax_temp, groups=inv_groups
        )

        self.expand = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn2    = nn.BatchNorm2d(channels)
        self.gamma  = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        y = self.act(self.bn1(self.reduce(x)))
        y = self.inv(y)
        y = self.bn2(self.expand(y))
        return self.act(x + self.gamma * y)
    
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
# UNet with Involution at bottleneck + SIMAM on skip connections
# ------------------------------------------------------------------
# Assumes you already defined:
# - class Involution(nn.Module): ...
# - class InvHead(nn.Module): ...
# from your provided code.

class UNetInvSimAM(nn.Module):
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

        # Bottleneck: use InvHead (which internally uses Involution)
        self.bottleneck = InvHead(
            channels=base_c * 16,
            reduce=4,
            k=7,               # kernel size for involution at bottleneck
            inv_reduction=4,
            kernel_norm="l2",
            softmax_temp=1.0,
            inv_groups=None,   # depthwise over hidden by default
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
    model = UNetInvSimAM(in_channels=1, num_classes=1, base_c=32)
    y = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)



