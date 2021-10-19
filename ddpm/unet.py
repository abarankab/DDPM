import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.normalization import GroupNorm


def get_norm(norm, num_channels, num_groups):
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")


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
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample(nn.Module):
    __doc__ = r"""Downsamples a given tensor by a factor of 2. Uses strided convolution. Assumes even height and width.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: ignored
        y: ignored
    Output:
        tensor of shape (N, in_channels, H // 2, W // 2)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()

        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
    
    def forward(self, x, time_emb, y):
        if x.shape[2] % 2 == 1:
            raise ValueError("downsampling tensor height should be even")
        if x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor width should be even")

        return self.downsample(x)


class Upsample(nn.Module):
    __doc__ = r"""Upsamples a given tensor by a factor of 2. Uses resize convolution to avoid checkerboard artifacts.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: ignored
        y: ignored
    Output:
        tensor of shape (N, in_channels, H * 2, W * 2)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )
    
    def forward(self, x, time_emb, y):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    __doc__ = r"""Applies QKV self-attention with a residual connection.
    
    Input:
        x: tensor of shape (N, in_channels, H, W)
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
    Output:
        tensor of shape (N, in_channels, H, W)
    Args:
        in_channels (int): number of input channels
    """
    def __init__(self, in_channels, norm="gn", num_groups=32):
        super().__init__()
        
        self.in_channels = in_channels
        self.norm = get_norm(norm, in_channels, num_groups)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)

        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)

        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b, h * w, h * w)

        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        assert out.shape == (b, h * w, c)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)

        return self.to_out(out) + x


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
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
        use_attention (bool): if True applies AttentionBlock to the output. Default: False
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout,
        time_emb_dim=None,
        num_classes=None,
        activation=F.relu,
        norm="gn",
        num_groups=32,
        use_attention=False,
    ):
        super().__init__()

        self.activation = activation

        self.norm_1 = get_norm(norm, in_channels, num_groups)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm_2 = get_norm(norm, out_channels, num_groups)
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.time_bias = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None
        self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None

        self.residual_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.attention = nn.Identity() if not use_attention else AttentionBlock(out_channels, norm, num_groups)
    
    def forward(self, x, time_emb=None, y=None):
        out = self.activation(self.norm_1(x))
        out = self.conv_1(out)

        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            out += self.time_bias(self.activation(time_emb))[:, :, None, None]

        if self.class_bias is not None:
            if y is None:
                raise ValueError("class conditioning was specified but y is not passed")

            out += self.class_bias(y)[:, :, None, None]

        out = self.activation(self.norm_2(out))
        out = self.conv_2(out) + self.residual_connection(x)
        out = self.attention(out)

        return out


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
        attention_resolutions (tuple): list of relative resolutions at which to apply attention. Default: ()
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
        initial_pad (int): initial padding applied to image. Should be used if height or width is not a power of 2. Default: 0
    """

    def __init__(
        self,
        img_channels,
        base_channels,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=None,
        time_emb_scale=1.0,
        num_classes=None,
        activation=F.relu,
        dropout=0.1,
        attention_resolutions=(),
        norm="gn",
        num_groups=32,
        initial_pad=0,
    ):
        super().__init__()

        self.activation = activation
        self.initial_pad = initial_pad

        self.num_classes = num_classes
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        ) if time_emb_dim is not None else None
    
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(
                    now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels
                channels.append(now_channels)
            
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)
        

        self.mid = nn.ModuleList([
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                activation=activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=True,
            ),
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                activation=activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=False,
            ),
        ])

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    channels.pop() + now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels
            
            if i != 0:
                self.ups.append(Upsample(now_channels))
        
        assert len(channels) == 0
        
        self.out_norm = get_norm(norm, base_channels, num_groups)
        self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)
    
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
        
        x = self.init_conv(x)

        skips = [x]

        for layer in self.downs:
            x = layer(x, time_emb, y)
            skips.append(x)
        
        for layer in self.mid:
            x = layer(x, time_emb, y)
        
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb, y)

        x = self.activation(self.out_norm(x))
        x = self.out_conv(x)
        
        if self.initial_pad != 0:
            return x[:, :, ip:-ip, ip:-ip]
        else:
            return x