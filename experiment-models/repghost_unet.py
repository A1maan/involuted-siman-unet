# repghost_unet.py
# U-Net with RepGhost blocks (PyTorch >= 1.10)
# Paper / repo (concept & fusion idea): RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization
# https://arxiv.org/abs/2211.06088  (module: add instead of concat; move ReLU; identity-BN branch; fuse for deploy)
# https://github.com/ChengpengChen/RepGhost

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utils
# -------------------------

def _bn_to_scale_shift(bn: nn.BatchNorm2d):
    """Return per-channel scale and shift for folding BN into preceding linear op."""
    # y = gamma * (x - mean) / sqrt(var + eps) + beta  ==  scale * x + shift
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    scale = gamma / torch.sqrt(var + eps)
    shift = beta - mean * scale
    return scale, shift


# -------------------------
# Squeeze-and-Excitation (optional)
# -------------------------

class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w


# -------------------------
# RepGhost module (training-time)  -> convertible to a single DW conv (deploy)
# Diagram matches Fig. 3(d-e) in the paper: depthwise conv branch + identity-BN branch, then ReLU.
# We assume Cin == Cout inside the module (preceded/followed by 1x1 convs in the bottleneck).
# -------------------------

class RepGhostModule(nn.Module):
    """
    Training graph:
        y = BN_dw(DW(x)) + BN_id(x)
        y = ReLU(y)
    Deploy graph (after convert_to_deploy):
        y = DW_fused(x) + bias; ReLU(y)
    """
    def __init__(self, channels: int, ksize: int = 3, stride: int = 1, deploy: bool = False):
        super().__init__()
        padding = ksize // 2
        self.channels = channels
        self.stride = stride
        self.ksize = ksize
        self.deploy = deploy

        if deploy:
            # Single depthwise conv with bias (fused)
            self.reparam = nn.Conv2d(channels, channels, ksize, stride, padding,
                                     groups=channels, bias=True)
            self.act = nn.ReLU(inplace=True)
        else:
            # Depthwise conv branch + BN
            self.dw = nn.Conv2d(channels, channels, ksize, stride, padding,
                                groups=channels, bias=False)
            self.dw_bn = nn.BatchNorm2d(channels)

            # Identity branch with BN (no spatial conv)
            self.id_bn = nn.BatchNorm2d(channels)

            self.act = nn.ReLU(inplace=True)

    @torch.no_grad()
    def convert_to_deploy(self):
        """Fuse BN_dw(DW) + BN_id into a single DW conv with bias."""
        if self.deploy:
            return

        # 1) Fold BN into depthwise conv weights/bias
        scale_dw, shift_dw = _bn_to_scale_shift(self.dw_bn)
        # dw conv has no bias:
        Wdw = self.dw.weight.clone()  # [C,1,kh,kw]
        # scale each channel's kernel by its scale_dw
        Wdw = Wdw * scale_dw.view(-1, 1, 1, 1)
        bdw = shift_dw.clone()  # [C]

        # 2) Convert BN_id(x) to a depthwise conv with an identity kernel
        scale_id, shift_id = _bn_to_scale_shift(self.id_bn)
        # Build an impulse (identity) kernel for depthwise conv
        k = torch.zeros_like(Wdw)  # [C,1,kh,kw]
        center = self.ksize // 2
        k[:, 0, center, center] = scale_id

        # 3) Sum both linear ops (same groups/channels), sum biases too
        W_fused = Wdw + k
        b_fused = bdw + shift_id

        # 4) Create reparam conv and load weights
        self.reparam = nn.Conv2d(self.channels, self.channels,
                                 self.ksize, self.stride, self.ksize // 2,
                                 groups=self.channels, bias=True)
        self.reparam.weight.data.copy_(W_fused)
        self.reparam.bias.data.copy_(b_fused)

        # 5) Cleanup training branches
        del self.dw, self.dw_bn, self.id_bn
        self.deploy = True

    def forward(self, x):
        if self.deploy:
            return self.act(self.reparam(x))
        else:
            y = self.dw_bn(self.dw(x))
            y = y + self.id_bn(x)
            return self.act(y)


# -------------------------
# RepGhost Bottleneck-ish block:
# 1x1 PW conv -> RepGhostModule -> (optional SE) -> 1x1 PW conv -> RepGhostModule
# Keeps Cin/Cout flexible (U-Net-style). Residual optional when Cin==Cout.
# -------------------------

class RGBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False, se_reduction=16, residual=False):
        super().__init__()
        mid = out_ch // 2  # "thinner" middle channels (Fig. 4b hint)
        mid = max(8, mid)

        self.proj1 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.rg1 = RepGhostModule(mid, ksize=3, stride=1)

        self.se = SqueezeExcite(mid) if use_se else nn.Identity()

        self.proj2 = nn.Sequential(
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # second RG
        self.rg2 = RepGhostModule(out_ch, ksize=3, stride=1)

        self.residual = residual and (in_ch == out_ch)

    def forward(self, x):
        identity = x
        x = self.proj1(x)
        x = self.rg1(x)
        x = self.se(x)
        x = self.proj2(x)
        x = self.rg2(x)
        if self.residual:
            x = x + identity
        return x

    @torch.no_grad()
    def convert_to_deploy(self):
        self.rg1.convert_to_deploy()
        self.rg2.convert_to_deploy()


# -------------------------
# U-Net with RepGhost blocks
# -------------------------

class DoubleRG(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False):
        super().__init__()
        self.b1 = RGBlock(in_ch, out_ch, use_se=use_se, residual=False)
        self.b2 = RGBlock(out_ch, out_ch, use_se=use_se, residual=True)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        return x

    @torch.no_grad()
    def convert_to_deploy(self):
        self.b1.convert_to_deploy()
        self.b2.convert_to_deploy()


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = DoubleRG(in_ch, out_ch, use_se=use_se)

    def forward(self, x):
        return self.block(self.pool(x))

    @torch.no_grad()
    def convert_to_deploy(self):
        self.block.convert_to_deploy()


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.reduce = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
            self.reduce = nn.Identity()
        self.block = DoubleRG(out_ch * 2, out_ch, use_se=use_se)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.reduce(x)
        # pad if needed (odd dims)
        diffY = skip.size(-2) - x.size(-2)
        diffX = skip.size(-1) - x.size(-1)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.block(x)

    @torch.no_grad()
    def convert_to_deploy(self):
        self.block.convert_to_deploy()


class RepGhostUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base_ch=32, use_se=False, bilinear=False):
        """
        base_ch=32 is a good lightweight start. Use 64 for larger models.
        """
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16

        self.inc   = DoubleRG(n_channels, c1, use_se=use_se)
        self.down1 = Down(c1, c2, use_se=use_se)
        self.down2 = Down(c2, c3, use_se=use_se)
        self.down3 = Down(c3, c4, use_se=use_se)
        self.down4 = Down(c4, c5, use_se=use_se)

        self.up1 = Up(c5, c4, use_se=use_se, bilinear=bilinear)
        self.up2 = Up(c4, c3, use_se=use_se, bilinear=bilinear)
        self.up3 = Up(c3, c2, use_se=use_se, bilinear=bilinear)
        self.up4 = Up(c2, c1, use_se=use_se, bilinear=bilinear)

        self.outc = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)     # [B, c1, H, W]
        x2 = self.down1(x1)  # [B, c2, H/2, W/2]
        x3 = self.down2(x2)  # [B, c3, H/4, W/4]
        x4 = self.down3(x3)  # [B, c4, H/8, W/8]
        x5 = self.down4(x4)  # [B, c5, H/16, W/16]

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        logits = self.outc(x)
        return logits

    @torch.no_grad()
    def convert_to_deploy(self):
        """Fuse all RepGhost modules in-place for faster inference."""
        for m in self.modules():
            if isinstance(m, DoubleRG):
                m.convert_to_deploy()
            elif isinstance(m, Down) or isinstance(m, Up):
                m.convert_to_deploy()
            elif isinstance(m, RGBlock):
                m.convert_to_deploy()


# -------------------------
# Quick sanity test
# -------------------------
if __name__ == "__main__":
    model = RepGhostUNet(n_channels=3, n_classes=1, base_ch=32, use_se=False, bilinear=False)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print("out:", y.shape)  # -> [1, 1, 256, 256]

    # Convert to deploy (after training + eval)
    model.eval()
    model.convert_to_deploy()
    with torch.no_grad():
        y2 = model(x)
    print("deploy out:", y2.shape)
