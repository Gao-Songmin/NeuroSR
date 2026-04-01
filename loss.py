import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ot  # POT: Python Optimal Transport


class Sobel3DLoss(nn.Module):
    def __init__(self, wx=1.0, wy=1.0, wz=2.0):
        super().__init__()
        self.wx = wx
        self.wy = wy
        self.wz = wz

        base_kernel = torch.tensor(
            [
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
            ],
            dtype=torch.float32,
        )

        kx = base_kernel.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
        ky = base_kernel.permute(1, 0, 2).unsqueeze(0).unsqueeze(0)
        kz = base_kernel.unsqueeze(0).unsqueeze(0)

        self.register_buffer("kernel_x", kx)
        self.register_buffer("kernel_y", ky)
        self.register_buffer("kernel_z", kz)

    def forward(self, pred, target):
        """
        pred, target: (B, 1, D, H, W)
        """
        gx_pred, gy_pred, gz_pred = self._compute_directional_grads(pred)
        gx_target, gy_target, gz_target = self._compute_directional_grads(target)

        loss_x = F.l1_loss(gx_pred, gx_target)
        loss_y = F.l1_loss(gy_pred, gy_target)
        loss_z = F.l1_loss(gz_pred, gz_target)

        return self.wx * loss_x + self.wy * loss_y + self.wz * loss_z

    def _compute_directional_grads(self, x):
        x = F.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate")

        gx = F.conv3d(x, self.kernel_x)
        gy = F.conv3d(x, self.kernel_y)
        gz = F.conv3d(x, self.kernel_z)

        return gx, gy, gz


class Sobel2DLoss(nn.Module):
    def __init__(self, wx=1.0, wy=2.0):
        """
        Sobel loss for 2D images. Computes L1 difference of x/y gradients.
        """
        super().__init__()
        self.wx = wx
        self.wy = wy

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )

        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )

        # Expand to shape (out_channels, in_channels, H, W) = (1,1,3,3)
        kx = sobel_x.view(1, 1, 3, 3)
        ky = sobel_y.view(1, 1, 3, 3)

        self.register_buffer("kernel_x", kx)
        self.register_buffer("kernel_y", ky)

    def forward(self, pred, target):
        """
        pred, target: Tensor of shape (B, 1, H, W)
        """
        gx_pred, gy_pred = self._compute_grads(pred)
        gx_target, gy_target = self._compute_grads(target)

        loss_x = F.l1_loss(gx_pred, gx_target)
        loss_y = F.l1_loss(gy_pred, gy_target)

        return self.wx * loss_x + self.wy * loss_y

    def _compute_grads(self, x):
        # Pad to preserve spatial size
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(x, self.kernel_x)
        gy = F.conv2d(x, self.kernel_y)
        return gx, gy


class HessianLoss3D(nn.Module):
    def __init__(
        self, charbonnier_eps: float | None = 1e-3, reduction: str | None = "mean"
    ):
        super().__init__()

        self.charbonnier_eps = charbonnier_eps
        self.reduction = reduction
        # kernels for second-order derivatives
        K2 = torch.tensor([1.0, -2.0, 1.0])
        self.K2x = K2.view(1, 1, 1, 1, 3)
        self.K2y = K2.view(1, 1, 1, 3, 1)
        self.K2z = K2.view(1, 1, 3, 1, 1)
        # kernels for first-order derivatives
        K1 = torch.tensor([-0.5, 0.0, 0.5])
        self.K1x = K1.view(1, 1, 1, 1, 3)
        self.K1y = K1.view(1, 1, 1, 3, 1)
        self.K1z = K1.view(1, 1, 3, 1, 1)

        self.pad_x = (0, 0, 1)
        self.pad_y = (0, 1, 0)
        self.pad_z = (1, 0, 0)

    def _calc_grads(self, vol: torch.Tensor) -> torch.Tensor:
        K2x = self.K2x.to(dtype=vol.dtype, device=vol.device)
        K2y = self.K2y.to(dtype=vol.dtype, device=vol.device)
        K2z = self.K2z.to(dtype=vol.dtype, device=vol.device)
        K1x = self.K1x.to(dtype=vol.dtype, device=vol.device)
        K1y = self.K1y.to(dtype=vol.dtype, device=vol.device)
        K1z = self.K1z.to(dtype=vol.dtype, device=vol.device)

        # --- second derivatives ---
        d_xx = F.conv3d(vol, K2x, padding=self.pad_x)
        d_yy = F.conv3d(vol, K2y, padding=self.pad_y)
        d_zz = F.conv3d(vol, K2z, padding=self.pad_z)

        # --- mixed derivatives ---
        d_xy = F.conv3d(F.conv3d(vol, K1x, padding=self.pad_x), K1y, padding=self.pad_y)
        d_xz = F.conv3d(F.conv3d(vol, K1x, padding=self.pad_x), K1z, padding=self.pad_z)
        d_yz = F.conv3d(F.conv3d(vol, K1y, padding=self.pad_y), K1z, padding=self.pad_z)

        return d_xx, d_yy, d_zz, d_xy, d_xz, d_yz

    def forward(self, pred, target):
        p_xx, p_yy, p_zz, p_xy, p_xz, p_yz = self._calc_grads(pred)
        t_xx, t_yy, t_zz, t_xy, t_xz, t_yz = self._calc_grads(target)

        l_xx = F.l1_loss(p_xx, t_xx)
        l_yy = F.l1_loss(p_yy, t_yy)
        l_zz = F.l1_loss(p_zz, t_zz)
        l_xy = F.l1_loss(p_xy, t_xy)
        l_xz = F.l1_loss(p_xz, t_xz)
        l_yz = F.l1_loss(p_yz, t_yz)

        loss = l_xx + l_yy + l_zz + l_xy + l_xz + l_yz
        return loss


class HessianPenalty3D(nn.Module):
    def __init__(
        self, charbonnier_eps: float | None = 1e-3, reduction: str | None = "mean"
    ):
        super().__init__()

        self.charbonnier_eps = charbonnier_eps
        self.reduction = reduction
        # kernels for second-order derivatives
        K2 = torch.tensor([1.0, -2.0, 1.0])
        self.K2x = K2.view(1, 1, 1, 1, 3)
        self.K2y = K2.view(1, 1, 1, 3, 1)
        self.K2z = K2.view(1, 1, 3, 1, 1)
        # kernels for first-order derivatives
        K1 = torch.tensor([-0.5, 0.0, 0.5])
        self.K1x = K1.view(1, 1, 1, 1, 3)
        self.K1y = K1.view(1, 1, 1, 3, 1)
        self.K1z = K1.view(1, 1, 3, 1, 1)

        self.pad_x = (0, 0, 1)
        self.pad_y = (0, 1, 0)
        self.pad_z = (1, 0, 0)

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        K2x = self.K2x.to(dtype=vol.dtype, device=vol.device)
        K2y = self.K2y.to(dtype=vol.dtype, device=vol.device)
        K2z = self.K2z.to(dtype=vol.dtype, device=vol.device)
        K1x = self.K1x.to(dtype=vol.dtype, device=vol.device)
        K1y = self.K1y.to(dtype=vol.dtype, device=vol.device)
        K1z = self.K1z.to(dtype=vol.dtype, device=vol.device)

        # --- second derivatives ---
        d_xx = F.conv3d(vol, K2x, padding=self.pad_x)
        d_yy = F.conv3d(vol, K2y, padding=self.pad_y)
        d_zz = F.conv3d(vol, K2z, padding=self.pad_z)

        # --- mixed derivatives ---
        d_xy = F.conv3d(F.conv3d(vol, K1x, padding=self.pad_x), K1y, padding=self.pad_y)
        d_xz = F.conv3d(F.conv3d(vol, K1x, padding=self.pad_x), K1z, padding=self.pad_z)
        d_yz = F.conv3d(F.conv3d(vol, K1y, padding=self.pad_y), K1z, padding=self.pad_z)

        if self.charbonnier_eps is None:
            penalty = (
                torch.abs(d_xx)
                + torch.abs(d_yy)
                + torch.abs(d_zz)
                + 2 * torch.abs(d_xy)
                + 2 * torch.abs(d_xz)
                + 2 * torch.abs(d_yz)
            )
        else:
            eps2 = self.charbonnier_eps**2
            penalty = (
                torch.sqrt(d_xx**2 + eps2)
                + torch.sqrt(d_yy**2 + eps2)
                + torch.sqrt(d_zz**2 + eps2)
                + torch.sqrt(d_xy**2 + eps2)
                + torch.sqrt(d_xz**2 + eps2)
                + torch.sqrt(d_yz**2 + eps2)
            )

        if self.reduction == "mean":
            return penalty.mean()
        elif self.reduction == "sum":
            return penalty.sum()
        elif self.reduction is None:
            return penalty
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


class WassersteinPointCloudLoss(nn.Module):
    """
    Compute Wasserstein (Earth Mover's) distance between two 3D volumes as sparse point clouds.
    Non-differentiable but useful for structural comparison or loss.
    """

    def __init__(self, threshold=1e-3, normalize_cost=True, default_value=1.0):
        """
        Args:
            threshold: minimum voxel value to include in point cloud
            normalize_cost: if True, normalize cost matrix to [0, 1]
            default_value: fallback value when point cloud is empty
        """
        super(WassersteinPointCloudLoss, self).__init__()
        self.threshold = threshold
        self.normalize_cost = normalize_cost
        self.default_value = default_value

    def forward(self, sr, hr):
        """
        sr, hr: Tensor of shape (B, 1, D, H, W)
        Returns: scalar Tensor (mean Wasserstein loss over batch)
        """
        losses = []
        sr_np = sr.detach().cpu().numpy()
        hr_np = hr.detach().cpu().numpy()

        B = sr.shape[0]
        for b in range(B):
            loss = self._emd_loss(sr_np[b, 0], hr_np[b, 0])
            losses.append(loss)

        return torch.tensor(losses, dtype=torch.float32, device=sr.device).mean()

    def _emd_loss(self, img1, img2):
        x_coords, x_weights = self._to_pointcloud(img1)
        y_coords, y_weights = self._to_pointcloud(img2)

        if len(x_coords) == 0 or len(y_coords) == 0:
            return self.default_value

        M = ot.dist(x_coords, y_coords, metric="euclidean")  # cost matrix

        if self.normalize_cost:
            M /= M.max()

        emd = ot.emd2(x_weights, y_weights, M)  # scalar EMD
        return emd

    def _to_pointcloud(self, img):
        """
        Convert 3D array to sparse point cloud (coords, weights)
        """
        mask = img > self.threshold
        coords = np.stack(np.nonzero(mask), axis=1).astype(np.float64)
        weights = img[mask].astype(np.float64)

        if weights.sum() == 0:
            return np.zeros((0, 3)), np.zeros((0,))

        weights /= weights.sum()
        return coords, weights


if __name__ == "__main__":
    x1 = torch.randn(2, 1, 64, 256, 256, requires_grad=True)  # (B,C,D,H,W)
    x2 = torch.randn(2, 1, 64, 256, 256, requires_grad=True)  # (B,C,D,H,W)
    l1_criterion = torch.nn.L1Loss()
    sobel_criterion = Sobel3DLoss()
    hess_criterion = HessianPenalty3D(charbonnier_eps=1e-3)

    l1_loss = l1_criterion(x1, x2)
    sobel_loss = sobel_criterion(x1, x2)
    hess_penalty = hess_criterion(x1)
    sparse_penalty = x1.abs().sum()

    print(f"L1 loss: {l1_loss}")
    print(f"Sobel loss: {0.05 * sobel_loss}")
    print(f"Hessian3D penalty:{0.2 * hess_penalty}")
    print(f"sparse_penalty:{2e-7 * sparse_penalty}")
