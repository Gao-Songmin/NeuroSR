import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndi


def wpsnr_torch(pred, gt, weight=None, data_range=1.0):
    _, _, H, W = pred.shape
    if weight is None:
        mse = torch.mean((pred - gt) ** 2)
    else:
        w = torch.clamp(weight.float(), min=0)
        mse = torch.sum(w * (pred - gt) ** 2) / torch.sum(w)
    return 10 * torch.log10((data_range**2) / mse)


def intensity_wpsnr(pred, gt, data_range=1.0):
    weights = gt / gt.max()
    intensity_wpsnr = wpsnr_torch(pred, gt, weights, data_range)
    return intensity_wpsnr


def gradient_wpsnr(pred: Tensor, gt: Tensor, data_range=1.0):
    def sobel_gradient_weight(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        assert x.size(1) == 1, "The input 2D image should have 1 channel."
        kx = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=x.dtype, device=x.device
        ).view(1, 1, 3, 3)
        ky = kx.transpose(-1, -2)
        x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(x_pad, kx)
        gy = F.conv2d(x_pad, ky)

        mag = torch.sqrt(gx**2 + gy**2 + eps)  # (N,C,H,W)
        w_map = mag.mean(dim=1, keepdim=True)  # (N,1,H,W) 作为权重
        return w_map

    w_map = sobel_gradient_weight(gt)
    intensity_wpsnr = wpsnr_torch(pred, gt, w_map, data_range)
    return intensity_wpsnr


def _laplace_9tap_conv(y: torch.Tensor) -> torch.Tensor:
    """
    9-tap Laplacian per Helmrich Eq.(16):
    h = 1/4 * (12c - 2*(N,S,E,W) - (NE,NW,SE,SW))
    边界使用 replicate padding（论文等价处理）。
    输入 y: (1,1,H,W)
    返回 h: (1,1,H,W)
    """
    k = (
        torch.tensor(
            [[-1, -2, -1], [-2, 12, -2], [-1, -2, -1]], dtype=y.dtype, device=y.device
        )
        * 0.25
    )
    k = k.view(1, 1, 3, 3)
    ypad = F.pad(y, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(ypad, k)


def wpsnr_helmrich(
    pred: torch.Tensor,
    gt: torch.Tensor,
    bit_depth: int = 8,
    block_size: int | None = 64,
    beta: float = 0.5,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict]:
    """
    计算 Helmrich 2019 的 WPSNR（2D，亮度通道）。
    参数
      pred, gt: (H,W)/(1,H,W)/(3,H,W)/(N,C,H,W) 中任意一种；可在 CPU/GPU；值域任意
      bit_depth: 图像位深（用于常量 a_min, a_pic 及峰值）
      block_size: 块大小；默认按分辨率选 64(≈HD) 或 128(≈UHD)
      beta: 论文经验值 0.5
    返回
      wpsnr (Tensor 标量), 以及一些中间量（dict）
    参考：Sec.2.1–2.2, Eq.(16–18):contentReference[oaicite:2]{index=2}
    """
    _, _, H, W = gt.shape

    # ---- 论文中的常量 ----
    a_min = 2 ** (bit_depth - 6) / 255.0
    a_pic = (3840.0 * 2160.0) / (W * H)

    # 若尺寸非块整数倍，padding
    pad_h = (block_size - (H % block_size)) % block_size
    pad_w = (block_size - (W % block_size)) % block_size
    if pad_h or pad_w:
        pad = (0, pad_w, 0, pad_h)  # left,right,top,bottom
        y_pred = F.pad(pred, pad, mode="replicate")
        y_gt = F.pad(gt, pad, mode="replicate")
        H2, W2 = H + pad_h, W + pad_w
    else:
        H2, W2 = H, W

    # ---- 拉普拉斯高通 -> 活动度 a_k ----
    h = _laplace_9tap_conv(y_gt)
    act_map = h.abs()

    # 计算块内平均
    k = block_size
    act_blk = (
        F.avg_pool2d(act_map, kernel_size=k, stride=k, ceil_mode=False) ** 2
    )  # (1,1, H2/k, W2/k)
    a_min_sq = a_min**2
    a_k = torch.maximum(
        act_blk, torch.tensor(a_min_sq, dtype=act_blk.dtype, device=act_blk.device)
    )

    # ---- 权重 w_k (Eq.18): w_k = (a_pic / a_k)^beta，beta=0.5 ----
    w_k = (a_pic / (a_k + eps)) ** beta  # (1,1, nBh, nBw)

    # ---- 块加权 SSE： sum_k w_k * sum_{(x,y) in Bk} (pred-gt)^2 ----
    err2 = (y_pred - y_gt) ** 2
    # 块内 SSE = 块内均值 * 块像素数
    sse_blk = F.avg_pool2d(err2, kernel_size=k, stride=k) * (k * k)  # (1,1,nBh,nBw)
    # 加权 SSE
    DwSSE_pic = torch.sum(w_k * sse_blk)

    # ---- WPSNR ----
    num = W * H
    wpsnr = 10.0 * torch.log10(
        torch.tensor(num, dtype=err2.dtype, device=err2.device) / (DwSSE_pic + eps)
    )

    return wpsnr
