import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class PositionalEmbedding(nn.Module):
    __doc__ = r"""Computes a positional embedding of timesteps.

    Input:
        x: tensor of shape (N)
    Output:
        tensor of shape (N, dim)
    Args:
        dim (int): embedding dimension
        scale (float): linear scale to be applied to timesteps. Default: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample(nn.Module):
    __doc__ = r"""Downsamples a given tensor by a factor of 2. Uses strided convolution. Assumes even height and width.

    Input:
        x: tensor of shape (N, in_channels, H, W)
    Output:
        tensor of shape (N, in_channels, H // 2, W // 2)
    Args:
        in_channels (int): number of input channels
        use_reflection_pad (bool): if True reflection pad is used, otherwise zero pad is used. Default: False
    """

    def __init__(self, in_channels, use_reflection_pad=False):
        super().__init__()

        self.downsample = nn.Conv2d(
            in_channels, in_channels, 3, stride=2,
            padding=1, padding_mode="zeros" if not use_reflection_pad else "reflect",
        )
    
    def forward(self, x):
        if x.shape[2] % 2 == 1:
            raise ValueError("downsampling tensor height should be even")
        if x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor width should be even")

        return self.downsample(x)


class Upsample(nn.Module):
    __doc__ = r"""Upsamples a given tensor by a factor of 2. Uses resize convolution to avoid checkerboard artifacts.

    Input:
        x: tensor of shape (N, in_channels, H, W)
    Output:
        tensor of shape (N, in_channels, H * 2, W * 2)
    Args:
        in_channels (int): number of input channels
        align_corners (bool): align_corners in bilinear upsampling. Default: True
        use_reflection_pad (bool): if True reflection pad is used, otherwise zero pad is used. Default: False
    """

    def __init__(self, in_channels, align_corners=True, use_reflection_pad=False):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=align_corners),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, padding_mode="zeros" if not use_reflection_pad else "reflect"),
        )
    
    def forward(self, x):
        return self.upsample(x)


class ConvBlock(nn.Module):
    __doc__ = r"""Applies 2d convolution, normalization and activation to a tensor.

    Input:
        x: tensor of shape (N, in_channels, H, W)
    Output:
        tensor of shape (N, out_channels, H, W)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        activation (function): activation function. Default: torch.nn.functional.relu
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "in"
        num_groups (int): number of groups used in group normalization. Default: 8
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        activation=F.relu,
        norm="in",
        num_groups=8,
    ):
        super().__init__()

        self.activation = activation

        modules = []
        modules.append(nn.Conv2d(
            in_channels, out_channels, 3, padding=1,
            bias=False if norm is not None else True,
        ))
        
        if norm == "in":
            modules.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif norm == "gn":
            modules.append(nn.GroupNorm(num_groups, out_channels))
        elif norm == "bn":
            modules.append(nn.BatchNorm2d(out_channels))
        elif norm is not None:
            raise ValueError("__init__() got unknown normalization type")
        
        self.block = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.activation(self.block(x))


class ResidualBlock(nn.Module):
    __doc__ = r"""Applies two conv blocks with resudual connection. Adds time and class conditioning by adding bias after first convolution.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: time embedding tensor of shape (N, time_emb_dim) or None if the block doesn't use time conditioning
        y: classes tensor of shape (N) or None if the block doesn't use class conditioning
    Output:
        tensor of shape (N, out_channels, H, W)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        time_emb_dim (int or None): time embedding dimension or None if the block doesn't use time conditioning. Default: None
        num_classes (int or None): number of classes or None if the block doesn't use class conditioning. Default: None
        activation (function): activation function. Default: torch.nn.functional.relu
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "in"
        num_groups (int): number of groups used in group normalization. Default: 8
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout,
        time_emb_dim=None,
        num_classes=None,
        **kwargs,
    ):
        super().__init__()

        self.conv_block_1 = ConvBlock(in_channels, out_channels, **kwargs)
        self.conv_block_2 = nn.Sequential(
            nn.Dropout(p=dropout),
            ConvBlock(out_channels, out_channels, **kwargs),
        )

        self.time_bias = nn.Sequential(nn.ReLU(), nn.Linear(time_emb_dim, out_channels)) if time_emb_dim is not None else None
        self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None

        self.residual_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, time_emb=None, y=None):
        out = self.conv_block_1(x)

        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            out += self.time_bias(time_emb)[:, :, None, None]

        if self.class_bias is not None:
            if y is None:
                raise ValueError("class conditioning was specified but y is not passed")

            out += self.class_bias(y)[:, :, None, None]
        
        out = self.conv_block_2(out)

        return out + self.residual_connection(x)


class AttentionBlock(nn.Module):
    __doc__ = r"""Applies linear attention with a residual connection.
    
    Input:
        x: tensor of shape (N, in_channels, H, W)
    Output:
        tensor of shape (N, in_channels, H, W)
    Args:
        in_channels (int): number of input channels
        heads (int): number of attention heads
        head_channels (int): number of channels in a head
    """
    def __init__(self, in_channels, heads=4, head_channels=32):
        super().__init__()
        self.heads = heads
        
        self.norm = nn.InstanceNorm2d(in_channels, affine=True)

        mid_channels = head_channels * heads
        self.to_qkv = nn.Conv2d(in_channels, mid_channels * 3, 1, bias=False)
        self.to_out = nn.Conv2d(mid_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(self.norm(x))
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out) + x


class UNet(nn.Module):
    __doc__ = """UNet model used to estimate noise.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: time embedding tensor of shape (N, time_emb_dim) or None if the block doesn't use time conditioning
        y: classes tensor of shape (N) or None if the block doesn't use class conditioning
    Output:
        tensor of shape (N, out_channels, H, W)
    Args:
        img_channels (int): number of image channels
        base_channels (int): number of base channels (after first convolution)
        channel_mults (tuple): tuple of channel multiplers. Default: (1, 2, 4, 8)
        time_emb_dim (int or None): time embedding dimension or None if the block doesn't use time conditioning. Default: None
        time_emb_scale (float): linear scale to be applied to timesteps. Default: 1.0
        num_classes (int or None): number of classes or None if the block doesn't use class conditioning. Default: None
        activation (function): activation function. Default: torch.nn.functional.relu
        dropout (float): dropout rate at the end of each residual block
        use_attn (bool): it True linear attention is used in residual blocks. Default: True
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "in"
        num_groups (int): number of groups used in group normalization. Default: 8
        align_corners (bool): align_corners in bilinear upsampling. Default: True
        use_reflection_pad (bool): if True reflection pad is used, otherwise zero pad is used. Default: False
        initial_pad (int): initial padding applied to image. Should be used if height or width is not a power of 2. Default: 0
    """

    def __init__(
        self,
        img_channels,
        base_channels,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=None,
        time_emb_scale=1.0,
        num_classes=None,
        activation=F.relu,
        dropout=0.1,
        use_attn=True,
        norm="in",
        num_groups=8,
        align_corners=True,
        use_reflection_pad=False,
        initial_pad=0,
    ):
        super().__init__()

        self.initial_pad = initial_pad

        self.num_classes = num_classes
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(time_emb_dim, time_emb_scale),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        ) if time_emb_dim is not None else None
    
        channels = (img_channels, *[base_channels * mult for mult in channel_mults])
        channel_pairs = tuple(zip(channels[:-1], channels[1:]))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (in_channels, out_channels) in enumerate(channel_pairs):
            is_last = (ind == len(channel_pairs) - 1)

            self.downs.append(nn.ModuleList([
                ResidualBlock(in_channels, out_channels, dropout,
                              time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation, norm=norm, num_groups=num_groups),
                ResidualBlock(out_channels, out_channels, dropout,
                              time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation, norm=norm, num_groups=num_groups),
                AttentionBlock(out_channels) if use_attn else nn.Identity(),
                Downsample(out_channels, use_reflection_pad=use_reflection_pad) if not is_last else nn.Identity(),
            ]))
        
        mid_channels = channels[-1]
        self.mid1 = ResidualBlock(
            mid_channels, mid_channels, dropout,
            time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation, norm=norm, num_groups=num_groups)
        self.mid_attn = AttentionBlock(mid_channels) if use_attn else nn.Identity()
        self.mid2 = ResidualBlock(
            mid_channels, mid_channels, dropout,
            time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation, norm=norm, num_groups=num_groups)

        for ind, (in_channels, out_channels) in enumerate(reversed(channel_pairs[1:])):
            self.ups.append(nn.ModuleList([
                ResidualBlock(out_channels * 2, in_channels, dropout,
                              time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation, norm=norm, num_groups=num_groups),
                ResidualBlock(in_channels, in_channels, dropout,
                              time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation, norm=norm, num_groups=num_groups),
                AttentionBlock(in_channels) if use_attn else nn.Identity(),
                Upsample(in_channels, align_corners=align_corners, use_reflection_pad=use_reflection_pad),
            ]))
        
        self.out_conv = nn.Sequential(
            ConvBlock(base_channels, base_channels, activation=activation, norm=norm, num_groups=num_groups),
            nn.Conv2d(base_channels, img_channels, 1),
        )
    
    def forward(self, x, time=None, y=None):
        ip = self.initial_pad
        if ip != 0:
            x = F.pad(x, (ip,) * 4)

        if self.time_mlp is not None:
            if time is None:
                raise ValueError("time conditioning was specified but tim is not passed")
            
            time_emb = self.time_mlp(time)
        else:
            time_emb = None
        
        if self.num_classes is not None and y is None:
            raise ValueError("class conditioning was specified but y is not passed")
        
        skips = []

        for r1, r2, attn, downsample in self.downs:
            x = r1(x, time_emb, y)
            x = r2(x, time_emb, y)
            x = attn(x)
            skips.append(x)
            x = downsample(x)
        
        x = self.mid1(x, time_emb, y)
        x = self.mid_attn(x)
        x = self.mid2(x, time_emb, y)
        
        for r1, r2, attn, upsample in self.ups:
            x = r1(torch.cat([x, skips.pop()], dim=1), time_emb, y)
            x = r2(x, time_emb, y)
            x = attn(x)
            x = upsample(x)
        
        if self.initial_pad != 0:
            return self.out_conv(x)[:, :, ip:-ip, ip:-ip]
        else:
            return self.out_conv(x)
