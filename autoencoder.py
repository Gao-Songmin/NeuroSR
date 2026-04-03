import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3dBlock(nn.Module):
    def __init__(self, cin, cout, k=(3, 3, 3)):
        super().__init__()
        pad = tuple(ki // 2 for ki in k)
        self.block = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=k, padding=pad, bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class Conv2dBlock(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        pad = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, padding=pad, bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DownSample(nn.Module):
    def __init__(self, channel, scale=(4, 1, 1)):
        super().__init__()
        self.pad = tuple(si // 2 for si in scale)
        kernel_size = [s // 2 * 2 + 1 for s in scale]
        self.conv = nn.Conv3d(
            channel,
            channel,
            kernel_size=kernel_size,
            stride=scale,
            padding=self.pad,
        )

    def forward(self, x):
        return self.conv(x)


############################################
# Encoder: 3D -> 2D + skip connection
############################################
class Encoder3DTo2D(nn.Module):
    """
    Input:  x3d [B, 1, 64, 64, 256]
    Output:  f2d [B, 64, 64, 256]
    """

    def __init__(self):
        super().__init__()
        self.down_sample1 = DownSample(channel=16)
        self.down_sample2 = DownSample(channel=32)
        self.down_sample3 = DownSample(channel=64)
        self.conv3d_l1 = nn.Sequential(
            Conv3dBlock(1, 4, k=(3, 3, 3)),
            Conv3dBlock(
                4,
                16,
                k=(3, 3, 3),
            ),
        )
        self.conv3d_l2 = Conv3dBlock(16, 32, k=(3, 3, 3))
        self.conv3d_l3 = Conv3dBlock(32, 64, k=(3, 3, 3))

    def forward(self, x3d):
        """
        x3d: (B, 1, 64, 64, 256)
        """
        if x3d.ndim == 4:
            x3d = x3d.unsqueeze(dim=1)
        f1 = self.conv3d_l1(x3d)  # (B, 16, 64, 64, 256)
        f2 = self.conv3d_l2(self.down_sample1(f1))  # (B, 32, 16, 64, 256)
        f3 = self.conv3d_l3(self.down_sample2(f2))  # (B, 64, 4, 64, 256)
        out = self.down_sample3(f3).squeeze(dim=2)  # (B, 64, 64, 256)

        skips = {"f3": f3}
        return out, skips


############################################
# Bottleneck：2D convolution
############################################
class BottleNeck2D(nn.Module):
    """
    预训练阶段用于替代 2D SR 模块
    输入:  [B, 64, 64, 256]
    输出:  [B, 128, 64, 256]
    """

    def __init__(self, in_ch=64, out_ch=128):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2dBlock(in_ch, 128, k=3),
            Conv2dBlock(128, out_ch, k=3),
        )

    def forward(self, x):
        return self.conv(x)


############################################
# Decoder: 2D -> 3D ConvTranspose3d + skip fusion
############################################
class Decoder2DTo3D(nn.Module):
    """
    输入:  f2d_bottleneck [B,128,64,256]
    skip_low:  [B,32,64,64,256]
    skip_high: [B,64,64,64,256]
    输出: [B,1,64,64,256]
    """

    def __init__(self, in2d_ch=128, out_ch=1, base_depth=4):
        super().__init__()
        self.base_depth = base_depth 

        self.up1 = nn.ConvTranspose3d(64, 64, kernel_size=(4, 1, 1), stride=(4, 1, 1))
        self.up2 = nn.ConvTranspose3d(32, 32, kernel_size=(4, 1, 1), stride=(4, 1, 1))

        self.conv_layer0 = nn.Sequential(
            Conv2dBlock(in2d_ch, in2d_ch, k=1),
            Conv2dBlock(in2d_ch, 64 * base_depth, k=1),
        )

        self.conv_layer1 = nn.Sequential(Conv3dBlock(64 + 64, 64, k=(3, 3, 3)))
        self.conv_layer2 = nn.Sequential(Conv3dBlock(64, 32, k=(3, 3, 3)))
        self.conv_layer3 = nn.Sequential(
            Conv3dBlock(32, 16, k=(3, 3, 3)),
            nn.Conv3d(16, 4, kernel_size=3, padding=1),
        )

    def forward(self, f2d, skips):
        B, C, H, W = f2d.shape
        f2d = self.conv_layer0(f2d)  # (B, 64*D0, H, W)
        x = f2d.reshape(B, -1, self.base_depth, H, W)  # (B, 64, 4, H, W)
        x = torch.concatenate([skips["f3"], x], dim=1)  # (B, 128, 4, H, W)
        x = self.up1(self.conv_layer1(x))  # (B, 64, 16, H, W)
        x = self.up2(self.conv_layer2(x))  # (B, 32, 64, H, W)
        out = self.conv_layer3(x)  # (B, 4, 64, H, W)
        # out = torch.mean(x, dim=1)
        return out


class AutoEncoder3Dto2Dto3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder3DTo2D()
        self.bottleneck = BottleNeck2D(in_ch=64, out_ch=128)
        self.decoder = Decoder2DTo3D(in2d_ch=128, out_ch=1, base_depth=4)

    def forward(self, x3d):
        """
        x3d: (B,1,64,64,256)
        """
        f2d, skips = self.encoder(x3d)  # -> (B,64,64,256)
        f2d_b = self.bottleneck(f2d)  # -> (B,128,64,256)
        out3d = self.decoder(f2d_b, skips)  # -> (B,1,64,64,256)
        return out3d
