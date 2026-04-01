import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import torch.utils.checkpoint as checkpoint
from torchvision.models import vgg19
# from basicsr.utils.registry import ARCH_REGISTRY

from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat

from autoencoder import *


def print_networks(net: nn.Module, verbose=True, log_file_path=None):
    num_params = 0
    message = "---------- Networks initialized -------------\n"
    for param in net.parameters():
        num_params += param.numel()
    if verbose:
        message += net.__str__() + "\n"
    message += "[Network] Total number of parameters : %.3f M\n" % (num_params / 1e6)
    message += "-----------------------------------------------"
    print(message)
    with open(log_file_path, "a") as log_file:
        log_file.write(message)


class Feature_Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.vgg19_54(x)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CASSDiff2(nn.Module):
    """Convolutional Across Scale Self-similarity, which add scale transform on CASS Layer"""

    def __init__(self, init_rp=0.0, align_corners=True, s_min=0.1):
        super().__init__()
        self.rp1 = nn.Parameter(torch.tensor(float(init_rp)))
        self.rp2 = nn.Parameter(torch.tensor(float(init_rp)))
        self.rp3 = nn.Parameter(torch.tensor(float(init_rp)))
        self.rp4 = nn.Parameter(torch.tensor(float(init_rp)))
        self.raw_s1 = nn.Parameter(torch.zeros(1))
        self.s_min = float(s_min)

        self.align_corners = align_corners

        self._cached_hw = None
        self._cached_base = None

    def _make_base_grid_rowcol(self, H, W, device, dtype):
        row = torch.linspace(-1.0, 1.0, steps=H, device=device, dtype=dtype)
        col = torch.linspace(-1.0, 1.0, steps=W, device=device, dtype=dtype)
        row_grid, col_grid = torch.meshgrid(row, col, indexing="ij")  # (H, W), # (H, W)
        base = torch.stack([row_grid, col_grid], dim=0)  # (2, H, W)
        return base

    def _base_grid(self, H, W, device, dtype):
        if (
            self._cached_hw != (H, W)
            or self._cached_base is None
            or self._cached_base.device != device
            or self._cached_base.dtype != dtype
        ):
            self._cached_hw = (H, W)
            self._cached_base = self._make_base_grid_rowcol(H, W, device, dtype)
        return self._cached_base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W)
        return diff2: (B,C,H,W)
        """
        B, C, H, W = x.shape
        base = self._base_grid(H, W, x.device, x.dtype)
        base_row = base[0]
        base_col = base[1]

        # scale in (0, 16)
        s_min, s_max = 1.0, 4.0
        s1 = s_min + (s_max - s_min) * torch.sigmoid(self.raw_s1)

        s1_row = base_row * s1 + 2.0 / float(H) * self.rp1
        s1_col = base_col * s1 + 2.0 / float(W) * self.rp2
        s2_row = base_row + 2.0 / float(H) * self.rp3
        s2_col = base_col + 2.0 / float(W) * self.rp4

        grid1 = torch.stack([s1_row, s1_col], dim=-1).expand(B, H, W, 2).contiguous()
        grid2 = torch.stack([s2_row, s2_col], dim=-1).expand(B, H, W, 2).contiguous()

        sft1 = F.grid_sample(
            x,
            grid1,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=self.align_corners,
        )
        sft2 = F.grid_sample(
            x,
            grid2,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=self.align_corners,
        )

        diff2 = (sft1 - sft2).pow(2)
        return diff2


class CASS(nn.Module):
    """Convolutional Across Scale Self-similarity"""

    def __init__(self, in_ch: int, mid_ch: int = None):
        super().__init__()
        if mid_ch is None:
            mid_ch = max(8, in_ch // 4)

        self.cass = CASSDiff2(init_rp=0.0)
        self.act = nn.GELU()

    def forward(self, x):
        diff = self.cass(x)
        cass = self.act(diff)
        return cass


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(
            self.d_state, self.d_inner, copies=4, merge=True
        )  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack(
            [
                x.view(B, -1, L),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),
            ],
            dim=1,
        ).view(B, 2, -1, L)
        xs = torch.cat(
            [x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1
        )  # (1, 4, 192, 3136)

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight
        )
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight
        )
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = (
            torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )
        invwh_y = (
            torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        expand: float = 2.0,
        use_CASS=True,
        is_light_sr: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim,
            d_state=d_state,
            expand=expand,
            dropout=attn_drop_rate,
            **kwargs,
        )
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        if use_CASS:
            self.cass = CASS(hidden_dim)
        else:
            self.cass = nn.Identity()
        self.mlp = Mlp(
            hidden_dim,
            int(hidden_dim * expand),  ## should transfer to integer!
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        # x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        # x = (
        #     self.cass(self.ln_2(x).permute(0, 3, 1, 2).contiguous())
        #     .permute(0, 2, 3, 1)
        #     .contiguous()
        # )
        # x = x * self.skip_scale2 + self.mlp(x)
        temp = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = (
            self.cass(self.ln_2(temp).permute(0, 3, 1, 2).contiguous())
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        x = temp * self.skip_scale2 + self.mlp(x)
        x = x.view(B, -1, C).contiguous()
        return x


class BasicLayer(nn.Module):
    """The Basic MambaIR Layer in one Residual State Space Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        drop_path=0.0,
        d_state=16,
        mlp_ratio=2.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        use_CASS=True,
        is_light_sr=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=nn.LayerNorm,
                    attn_drop_rate=0,
                    d_state=d_state,
                    expand=self.mlp_ratio,
                    input_resolution=input_resolution,
                    use_CASS=use_CASS,
                    is_light_sr=is_light_sr,
                )
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class ResidualGroup(nn.Module):
    """Residual State Space Group (RSSG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        d_state=16,
        mlp_ratio=4.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        img_size=None,
        patch_size=None,
        resi_connection="1conv",
        is_light_sr=False,
        use_CASS=True,
    ):
        super(ResidualGroup, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution  # [64, 64]

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            d_state=d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            use_CASS=use_CASS,
            is_light_sr=is_light_sr,
        )

        # build the last conv layer in each residual state space group
        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

    def forward(self, x, x_size):
        return (
            self.patch_embed(
                self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))
            )
            + x
        )

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class MambaIR_3Dto2D(nn.Module):
    r"""MambaIR Model
        A PyTorch impl of : `A Simple Baseline for Image Restoration with State Space Model `.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        d_state (int): num of hidden state in the state space model. Default: 16
        depths (tuple(int)): Depth of each RSSG
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4 for image SR, 1 for denoising
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(
        self,
        img_size=256,
        patch_size=1,
        in_chans=64,
        embed_dim=128,
        depths=(2, 2, 2, 2, 2),
        drop_rate=0.0,
        d_state=16,
        mlp_ratio=2.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        upscale=4,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        use_CASS=True,
        **kwargs,
    ):
        super(MambaIR_3Dto2D, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 4
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio = mlp_ratio
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.encoder = Encoder3DTo2D()
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = True if self.upsampler == "pixelshuffledirect" else False
        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build Residual State Space Group (RSSG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  # 6-layer
            layer = ResidualGroup(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                d_state=d_state,
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                is_light_sr=self.is_light_sr,
                use_CASS=use_CASS,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in the end of all residual groups
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        # -------------------------3. high-quality image reconstruction ------------------------ #
        self.decoder = Decoder2DTo3D()
        if self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv3d(4, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.upsample = Upsample3D(upscale, num_feat)
            self.conv_last = nn.Conv3d(1, 1, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)

        else:
            # for image denoising
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)  # N,L,C

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        # self.mean = self.mean.type_as(x)
        # x = (x - self.mean) * self.img_range

        if self.upsampler == "pixelshuffle":
            # for classical SR
            in2d, skip = self.encoder(x)
            x = self.conv_first(in2d)
            out2d = self.conv_after_body(self.forward_features(x)) + x
            y = self.decoder(out2d, skip)
            y = self.conv_before_upsample(y)
            out = self.conv_last(self.upsample(y))

        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR
            in2d = self.encoder(x)
            x = self.conv_first(in2d)
            out2d = self.conv_after_body(self.forward_features(x)) + x
            y = self.decoder(out2d)
            out = self.upsample(y)

        else:
            # for image denoising
            in2d = self.encoder(x)
            x_first = self.conv_first(in2d)
            out2d = self.conv_after_body(self.forward_features(x_first)) + x_first
            y = self.decoder(out2d)
            out = x + self.conv_last(y)

        # x = x / self.img_range + self.mean

        return out.squeeze(dim=1), {"in2d": in2d, "out2d": out2d}  # B, D, H, W

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r"""transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r"""return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(
            x.shape[0], self.embed_dim, x_size[0], x_size[1]
        )  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops


# class Vertical_Upscale(nn.Module):
#     def __init__(self, upscale_factor=2):
#         super().__init__()
#         self.scale_factor = upscale_factor
#     def forward(self, x):
#         b, c, d, h, w = x.shape
#         r = self.scale_factor
#         x =


class VerticalPixelShuffel3D(nn.Module):
    def __init__(self, upscale_factor=2):
        super().__init__()
        self.scale_factor = upscale_factor

    def forward(self, x):
        b, c, d, h, w = x.shape
        r = self.scale_factor
        x = x.view(b, c // r, r, d, h, w)  # (B, C//r, r, D, H, W)
        x = x.permute(0, 1, 3, 4, 2, 5)  # (B, C//r, D, H, r, W)
        x = x.reshape(b, c // r, d, h * r, w)
        return x


class VerticalPixelShuffel(nn.Module):
    def __init__(self, upscale_factor=2):
        super().__init__()
        self.scale_factor = upscale_factor

    def forward(self, x):
        b, c, h, w = x.shape
        r = self.scale_factor
        x = x.view(b, c // r, r, h, w)  # (B, C//r, r, H, W)
        x = x.permute(0, 1, 3, 2, 4)  # (B, C//r, H, r, W)
        x = x.reshape(b, c // r, h * r, w)
        return x


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        # m.append(nn.PixelShuffle(scale))
        m.append(VerticalPixelShuffel(scale))
        super(UpsampleOneStep, self).__init__(*m)


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Convd(num_feat, 2 * num_feat, 3, 1, 1))
                # m.append(nn.PixelShuffle(2))
                m.append(VerticalPixelShuffel(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            # m.append(nn.PixelShuffle(3))
            m.append(VerticalPixelShuffel(3))
        else:
            raise ValueError(
                f"scale {scale} is not supported. Supported scales: 2^n and 3."
            )
        super(Upsample, self).__init__(*m)


class Upsample3D(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(VerticalPixelShuffel3D(2))
                m.append(nn.Conv3d(num_feat // 2, num_feat // 2, 3, 1, 1))
                num_feat = num_feat // 2
        elif scale == 3:
            m.append(VerticalPixelShuffel3D(3))
            m.append(nn.Conv3d(num_feat, num_feat, 3, 1, 1))
        else:
            raise ValueError(
                f"scale {scale} is not supported. Supported scales: 2^n and 3."
            )
        super(Upsample3D, self).__init__(*m)


# class CRB_Layer(nn.Module):
#     def __init__(self, nf1):
#         super(CRB_Layer, self).__init__()

#         self.mambaBlock = VSSBlock(hidden_dim=128)

#     def forward(self, x):
#         B, C, H, W = x.shape

#         x = x.permute(0, 2, 3, 1).view(B, H * W, C)

#         out = self.mambaBlock(x, (H, W))

#         out = out.view(B, H, W, C).permute(0, 3, 1, 2)

#         return out

# class UpsampleBlock(nn.Module):
#     def __init__(self, filters, upscale_factor=2):
#         super().__init__()
#         layers = [
#             nn.Conv2d(
#                 filters, filters, kernel_size=3, stride=1, padding=1, groups=filters
#             ),
#             nn.Conv2d(filters, filters * upscale_factor, kernel_size=1),
#             nn.LeakyReLU(inplace=True),
#             VerticalPixelShuffel(upscale_factor=upscale_factor),
#         ]
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)


# class Restorer(nn.Module):
#     def __init__(self, in_nc=64, out_nc=64, nf=128, nb=8, n_upsampler=2):
#         super(Restorer, self).__init__()
#         self.num_blocks = nb

#         self.head = nn.Conv2d(in_nc, nf, 3, stride=1, padding=1)

#         self.neuroTreeConv = NeuronTreeConv(nf)

#         self.dwc3x3 = nn.Conv2d(nf, nf, 3, stride=1, padding=1, groups=nf)

#         self.fuse_head = nn.Conv2d(2 * nf, nf, 1, 1)

#         body = [CRB_Layer(nf) for _ in range(nb)]
#         self.body = nn.Sequential(*body)

#         self.conv = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)

#         self.upsampling = nn.Sequential(
#             *[UpsampleBlock(nf, upscale_factor=2) for _ in range(n_upsampler)]
#         )

#         self.fusion = nn.Conv2d(nf, out_nc, 3, 1, 1)

#     def forward(self, input):
#         f_shallow = self.head(input)

#         # skeleton在空间上特别稀疏，因此，刚开始使用树状空洞卷积去做
#         # f1 = self.neuroTreeConv(f_shallow)
#         # f2 = self.dwc3x3(f)
#         # f = self.fuse_head(torch.cat([f1, f2], dim=1))

#         f = self.body(f_shallow)

#         f = self.conv(f)

#         out = torch.add(f, f_shallow)

#         out = self.upsampling(out)

#         out = self.fusion(out)

#         # 该实验使用clamp
#         #        out = torch.clamp(out, min=self.min, max=self.max)

#         return out


# class DAN(nn.Module):
#     def __init__(
#         self,
#         in_channels=64,
#         filters=128,
#         out_channels=64,
#         num_blocks=20,
#         min=0.0,
#         max=1.0,
#     ):
#         super(DAN, self).__init__()

#         self.Restorer = Restorer(
#             in_channels=in_channels,
#             filters=filters,
#             out_channels=out_channels,
#             num_blocks=num_blocks,
#         )
#         self.min = min
#         self.max = max
#         # 添加 Sigmoid 层
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, skeleton):
#         B, C, H, W = skeleton.shape

#         out = self.Restorer(skeleton)

#         # out = self.sigmoid(out)

#         return out


class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch=64, num_feat=256, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


# class NeuronTreeConv(nn.Module):
#     def __init__(self, channel=64):
#         super(NeuronTreeConv, self).__init__()

#         # 分多少个组
#         self.chunks_num = 8

#         # 每个组多少feature map
#         self.feature_num = channel // self.chunks_num

#         self.d6dwc_9x9_fw = nn.Conv2d(
#             in_channels=self.feature_num,
#             out_channels=self.feature_num,
#             kernel_size=9,
#             groups=self.feature_num,
#             padding=((9 // 2) * 6),
#             dilation=6,
#         )

#         self.d4dwc_9x9_fw = nn.Conv2d(
#             in_channels=self.feature_num,
#             out_channels=self.feature_num,
#             kernel_size=9,
#             groups=self.feature_num,
#             padding=((9 // 2) * 4),
#             dilation=4,
#         )

#         self.d2dwc_9x9_fw = nn.Conv2d(
#             in_channels=self.feature_num,
#             out_channels=self.feature_num,
#             kernel_size=9,
#             groups=self.feature_num,
#             padding=((9 // 2) * 2),
#             dilation=2,
#         )

#         self.conv_3x3 = nn.Conv2d(
#             self.feature_num * 2, self.feature_num * 2, 3, stride=1, padding=1
#         )

#         self.d2dwc_9x9_bw = nn.Conv2d(
#             in_channels=self.feature_num,
#             out_channels=self.feature_num,
#             kernel_size=9,
#             groups=self.feature_num,
#             padding=((9 // 2) * 2),
#             dilation=2,
#         )

#         self.d4dwc_9x9_bw = nn.Conv2d(
#             in_channels=self.feature_num,
#             out_channels=self.feature_num,
#             kernel_size=9,
#             groups=self.feature_num,
#             padding=((9 // 2) * 4),
#             dilation=4,
#         )

#         self.d6dwc_9x9_bw = nn.Conv2d(
#             in_channels=self.feature_num,
#             out_channels=self.feature_num,
#             kernel_size=9,
#             groups=self.feature_num,
#             padding=((9 // 2) * 6),
#             dilation=6,
#         )

#         self.conv1x1 = nn.Conv2d(channel, channel, 1, 1)

#         self.fusion = nn.Conv2d(channel * 2, channel, 1, 1)

#     def forward(self, x):
#         N, C, H, W = x.shape

#         x_channel = self.conv1x1(x)

#         x_chunks = torch.chunk(x, chunks=self.chunks_num, dim=1)

#         # print(x_chunks[0].shape)

#         x_group = []

#         # 从前往后，对每一组内的feature map做卷积
#         # 第i组
#         for i in range(self.chunks_num):
#             if i == 0:
#                 x_group.append(self.d6dwc_9x9_fw(x_chunks[i]))
#                 # print(x_group[i].shape)
#             if i == 1:
#                 x_group.append(self.d4dwc_9x9_fw(x_chunks[i]))
#                 # print(x_group[i].shape)
#             if i == 2:
#                 x_group.append(self.d2dwc_9x9_fw(x_chunks[i]))
#                 # print(x_group[i].shape)

#             if i == 3:
#                 x_group.append(
#                     self.conv_3x3(torch.cat([x_chunks[3], x_chunks[4]], dim=1))
#                 )
#                 # print(x_group[i].shape)
#             if i == 4:
#                 pass
#                 # x_group.append(self.dwc_3x3_2(x_chunks[i]))
#                 # # print(x_group[i].shape)

#             if i == 5:
#                 x_group.append(self.d2dwc_9x9_bw(x_chunks[i]))
#                 # print(x_group[i].shape)
#             if i == 6:
#                 x_group.append(self.d4dwc_9x9_bw(x_chunks[i]))
#                 # print(x_group[i].shape)
#             if i == 7:
#                 x_group.append(self.d6dwc_9x9_bw(x_chunks[i]))
#                 # print(x_group[i].shape)

#         x_tree = torch.cat(x_group, dim=1)

#         x = self.fusion(torch.cat([x_tree, x_channel], dim=1))

#         # print(x.shape)
#         return x
