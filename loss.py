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
