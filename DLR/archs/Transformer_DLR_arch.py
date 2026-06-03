## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from einops.layers.torch import Rearrange

from basicsr.utils.registry import ARCH_REGISTRY


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# w/o shape
class LayerNorm_Without_Shape(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm_Without_Shape, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return self.body(x)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
## Proposed in Restormer
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, embed_dim, group):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # prior
        if group == 1:
            self.ln1 = nn.Linear(embed_dim*4, dim)
            self.ln2 = nn.Linear(embed_dim*4, dim)

    def forward(self, x, prior=None):
        if prior is not None:
            k1 = self.ln1(prior).unsqueeze(-1).unsqueeze(-1)
            k2 = self.ln2(prior).unsqueeze(-1).unsqueeze(-1)
            x = (x * k1) + k2

        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
## Standard channel-based Attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, embed_dim, group):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # prior
        if group == 1:
            self.ln1 = nn.Linear(embed_dim*4, dim)
            self.ln2 = nn.Linear(embed_dim*4, dim)

    def forward(self, x, prior=None):
        b,c,h,w = x.shape
        if prior is not None:
            k1 = self.ln1(prior).unsqueeze(-1).unsqueeze(-1)
            k2 = self.ln2(prior).unsqueeze(-1).unsqueeze(-1)
            x = (x * k1) + k2 ## Similar to SPADE

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        ## q : c x (hw) k : (hw x c)
        ## attn: c x c
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
## Multi-DConv Head Transposed Self-Attention (MDTA)
## Standard channel-based Cross-Attention
class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNorm_type):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.norm = LayerNorm(dim, LayerNorm_type)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x_A, x_B):
        b,c,h,w = x_A.shape
        _X_A = x_A
        x_A = self.norm(x_A)
        x_B = self.norm(x_B)
        q = self.q_dwconv(self.q(x_A))
        kv = self.kv_dwconv(self.kv(x_B))
        k,v = kv.chunk(2, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        ## q : c x (hw) k : (hw x c)
        ## attn: c x c
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return _X_A + out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return x
    

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class BAGModule(nn.Module):
    """
    Brightness Adaptive Gate (BAG)
    ----------------------------------------------------------
    x_g = (1 - w) ⊙ x + w ⊙ x_norm
    where x_norm is per-instance, per-channel normalization
    and w = G(α) = α^2 / (α^2 + eps), produced by a GAP→Conv→ReLU→Conv router.

    Args:
        in_channels (int): number of channels in input feature map x (B, C, H, W)
        hidden_channels (int): channels inside the gating MLP (after GAP). If None, uses C//4 (min 1)
        eps_norm (float): epsilon for normalization std
        eps_gate (float): epsilon for the binarization function G(·)
        use_bias (bool): whether to use bias in 1x1 convs inside the router
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = None,
        eps_norm: float = 1e-5,
        eps_gate: float = 1e-6,
        use_bias: bool = True,
    ):
        super().__init__()
        C = in_channels
        H = hidden_channels if hidden_channels is not None else max(1, C // 4)

        # Learnable affine params for normalization: γ, β ∈ R^C
        self.gamma = nn.Parameter(torch.ones(1, C, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.eps_norm = eps_norm
        self.eps_gate = eps_gate

        # Dynamic gating router: GAP → Conv → ReLU → Conv → α
        # (conv 1x1 is used because the input after GAP is (B, C, 1, 1))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.router = nn.Sequential(
            nn.Conv2d(C, H, kernel_size=1, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(H, C, kernel_size=1, bias=use_bias),
        )

    @staticmethod
    def _binarize_like_g(alpha: torch.Tensor, eps: float) -> torch.Tensor:
        # G(α) = α^2 / (α^2 + ε), smoothly pushes values toward {0, 1}
        alpha2 = alpha * alpha
        return alpha2 / (alpha2 + eps)

    def _brightness_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise, instance-wise normalization:
        # x_norm = γ * (x - μ) / σ + β, where μ, σ computed over spatial dims (H, W) for each channel & instance
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps_norm)
        x_hat = (x - mean) / std
        return self.gamma * x_hat + self.beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            x_g: Tensor of shape (B, C, H, W)
        """
        # 1) brightness normalization (Eq. 1)
        x_norm = self._brightness_normalize(x)

        # 2) dynamic gate (router) to get α, then w = G(α) (Eq. 3)
        gap_feats = self.gap(x)                 # (B, C, 1, 1)
        alpha = self.router(gap_feats)          # (B, C, 1, 1)
        w = self._binarize_like_g(alpha, self.eps_gate)  # (B, C, 1, 1), in [0, 1]

        # 3) mix normalized vs original per channel (Eq. 2)
        x_g = (1.0 - w) * x + w * x_norm
        return x_g

from typing import Tuple, Optional

class ConvAct(nn.Module):
    """Conv + Activation"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act="gelu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        if act.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act.lower() == "leakyrelu":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))

class FineTuningUnit(nn.Module):
    """
    细粒度“卷积+激活”堆叠模块（对应图示中的 Fine-tuning Unit）
    结构： [Conv+Act] x num_layers  + 残差
    """
    def __init__(self, channels: int, num_layers: int = 3, act: str = "gelu"):
        super().__init__()
        blocks = []
        for _ in range(num_layers):
            blocks.append(ConvAct(channels, channels, k=3, s=1, p=1, act=act))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        # 残差微调
        return x + self.blocks(x)

class MaskHead(nn.Module):
    """
    从特征生成 K 类掩码的 logits（未做 softmax）
    输出形状： (B, K, H, W)
    """
    def __init__(self, in_ch: int, hidden: int = 64, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden, num_classes, 1, 1, 0)  # K 类 logits
        )

    def forward(self, x):
        return self.net(x)  # (B, K, H, W)

class DenoiseModule(nn.Module):
    """
    多类别掩码去噪模块
    - 掩码取值 {0,1,2,3,4}，其中 0 表示不处理，1..4 分别对应不同的 FineTuningUnit
    - hard_mask=True：使用 argmax 得到整型掩码（不可导），再 one-hot 合成
    - hard_mask=False：对 logits 做 softmax 概率加权混合（可导）

    输入:  x  (B,C,H,W)
    输出:  y  (B,C,H,W)  — 去噪/微调后的特征
          mask_idx (B,1,H,W) — 硬掩码下返回 0..K-1 的整型；软掩码下返回 None
          mask_prob (B,K,H,W) — 软掩码下返回每类概率；硬掩码下返回 one-hot（float）
    """
    def __init__(
        self,
        channels: int,
        unit_layers: int = 3,
        act: str = "gelu",
        hard_mask: bool = True,
        num_classes: int = 5,
        mask_hidden: int = 64,
        softmax_temp: float = 1.0
    ):
        super().__init__()
        assert num_classes >= 2, "num_classes 至少为 2（含一个不处理的 0 类 + 至少一个处理类）"
        self.hard_mask = hard_mask
        self.num_classes = num_classes
        self.softmax_temp = softmax_temp

        # 1) 生成 K 类掩码 logits
        self.mask_head = MaskHead(in_ch=channels, hidden=mask_hidden, num_classes=num_classes)

        # 2) 为类 1..K-1 准备各自的 FineTuningUnit；类 0 为“恒等”
        self.refiners = nn.ModuleList([
            FineTuningUnit(channels=channels, num_layers=unit_layers, act=act)
            for _ in range(num_classes - 1)
        ])

        # 3) 可选的前置层（与原始一致，默认恒等）
        self.pre = nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        returns:
          y: (B,C,H,W)
          mask_idx: (B,1,H,W) or None（软掩码时 None）
          mask_prob: (B,K,H,W) — 概率或 one-hot
        """
        B, C, H, W = x.shape

        # logits -> (B, K, H, W)
        logits = self.mask_head(x)

        if self.hard_mask:
            # 整型掩码：argmax 得到 (B, H, W)
            mask_idx_hw = torch.argmax(logits, dim=1)  # 0..K-1
            # one-hot -> (B, K, H, W)
            mask_onehot = torch.nn.functional.one_hot(mask_idx_hw, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            mask_prob = mask_onehot  # 作为权重使用（0/1）
            mask_idx = mask_idx_hw.unsqueeze(1).to(x.dtype)  # (B,1,H,W) 便于与原接口兼容
        else:
            # 软掩码：概率可导
            mask_prob = torch.softmax(logits / self.softmax_temp, dim=1)  # (B, K, H, W)
            mask_idx = None

        # 前置
        x_in = self.pre(x)

        # 类 0：恒等（原特征）
        w0 = mask_prob[:, 0:1, :, :]  # (B,1,H,W)
        y = x_in * w0

        # 类 1..K-1：分别通过各自的 FineTuningUnit
        # 计算效率：每个分支只跑一次，再用对应权重融合
        for k in range(1, self.num_classes):
            wk = mask_prob[:, k:k+1, :, :]  # (B,1,H,W)
            x_refined_k = self.refiners[k - 1](x_in)  # (B,C,H,W)
            y = y + x_refined_k * wk

        return y, mask_idx, mask_prob




class Prior_Fusion(nn.Module):
    def __init__(self, dim, num_heads, bias, embed_dim, group):
        super(Prior_Fusion, self).__init__()
        # self.BA = BAGModule(dim)
        self.num_heads = num_heads
        self.ca = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear(group*group, 1),
            Rearrange('b c n -> b n c'),
            nn.Linear(embed_dim*4, dim*2),
            Rearrange('b n c -> b c n'),
        )
        self.sa = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.denoise_block1 = DenoiseModule(channels=32, unit_layers=3, act="gelu", hard_mask=True)
        # self.denoise_block2 = DenoiseModule(channels=64, unit_layers=3, act="gelu", hard_mask=True)
        # self.denoise_block3 = DenoiseModule(channels=128, unit_layers=3, act="gelu", hard_mask=True)
        # self.denoise_block4 = DenoiseModule(channels=256, unit_layers=3, act="gelu", hard_mask=True)
        
    def forward(self, x_A, x_B, prior=None):
        # x_B = self.BA(x_B)
        ca  =self.ca(prior).unsqueeze(-1)
        ca_A, ca_B = ca.chunk(2, dim=1)
        sa_A = self.sa(x_A)
        sa_B = self.sa(x_B)
        x = self.sigmoid(ca_A *  sa_A) * x_A + self.sigmoid(ca_B * sa_B) * x_B
        out = self.project_out(x)
        # print(out.shape)
        # if out.size(1) == 32:
        #     out, _, _ = self.denoise_block1(out)
        # elif out.size(1) == 64:
        #     out, _, _ = self.denoise_block2(out)
        # elif out.size(1) == 128:
        #     out, _, _ = self.denoise_block3(out)
        # else:
        #     out, _, _ = self.denoise_block4(out)
        return out

class SPMM(nn.Module):
    ## Semantic Prior Modulation module
    def __init__(self, dim, num_heads, bias, embed_dim, LayerNorm_type, qk_scale=None):
        super(SPMM, self).__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.norm1 = LayerNorm_Without_Shape(dim, LayerNorm_type)
        self.norm2 = LayerNorm_Without_Shape(embed_dim*4, LayerNorm_type)

        self.q = nn.Linear(dim, dim, bias=bias)
        self.kv = nn.Linear(embed_dim*4, 2*dim, bias=bias)
        
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, prior):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        _x = self.norm1(x)
        prior = self.norm2(prior)
        
        q = self.q(_x)
        kv = self.kv(prior)
        k,v = kv.chunk(2, dim=-1)   

        q = rearrange(q, 'b n (head c) -> b head n c', head=self.num_heads)
        k = rearrange(k, 'b n (head c) -> b head n c', head=self.num_heads)
        v = rearrange(v, 'b n (head c) -> b head n c', head=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head n c -> b n (head c)', head=self.num_heads)
        out = self.proj(out)

        # sum
        x = x + out
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()

        return x

class DPMM(nn.Module):
    ## Degradation Prior modulation module
    def __init__(self, dim, num_heads, bias, embed_dim, LayerNorm_type, group=1):
        super(DPMM, self).__init__()

        self.num_heads = num_heads
        self.group = group
        self.weight_linear = nn.Linear(dim, group * group)
        
        self.prior_norm = LayerNorm_Without_Shape(embed_dim*4, LayerNorm_type)
        self.prior_linear = nn.Linear(embed_dim*4, 2*dim, bias=bias)
        
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        
    def forward(self, x, prior):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prior_weights = F.softmax(self.weight_linear(emb), dim=1)
        prior = prior * prior_weights.unsqueeze(-1)
        prior = torch.sum(prior, dim=1)
        params = self.prior_linear(self.prior_norm(prior))
        alpha, beta = params.chunk(2, dim=-1)
        x = x * alpha.unsqueeze(-1).unsqueeze(-1) + beta.unsqueeze(-1).unsqueeze(-1)
        x = self.conv3x3(x)
        return x

##########################################################################
## Hierarchical Integration Module
class HIM(nn.Module):
    def __init__(self, dim, num_heads, bias, embed_dim, LayerNorm_type, qk_scale=None, group=None):
        super(HIM, self).__init__()
        self.group = group
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.norm1 = LayerNorm_Without_Shape(dim, LayerNorm_type)
        self.norm2_A = LayerNorm_Without_Shape(embed_dim*4, LayerNorm_type)
        self.down_A = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear(group*group, 1),
            Rearrange('b c n -> b n c'),
        )
        self.norm2_B = LayerNorm_Without_Shape(embed_dim*4, LayerNorm_type)
        self.down_B = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear(group*group, 1),
            Rearrange('b c n -> b n c'),
        )

        self.q = nn.Linear(dim, dim, bias=bias)
        ## 为什么embed_dim*4 要×4
        self.kv_A = nn.Linear(embed_dim*4, 2*dim, bias=bias)        
        self.proj_A = nn.Linear(dim, dim, bias=True)
        
        self.kv_B = nn.Linear(embed_dim*4, 2*dim, bias=bias)        
        self.proj_B = nn.Linear(dim, dim, bias=True)
        
        ## feature modulation
        self.kernel_A = nn.Sequential(
            nn.Linear(embed_dim*4, dim*2, bias=False),
        )
        self.kernel_B = nn.Sequential(
            nn.Linear(embed_dim*4, dim*2, bias=False),
        )
        
        
    def forward(self, x, prior_A, prior_B):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        _x = self.norm1(x)
        prior_A = self.norm2_A(prior_A)
        prior_B = self.norm2_B(prior_B)
        Parms_A = self.down_A(prior_A)
        Parms_A = self.kernel_A(Parms_A.squeeze()).view(-1, C * 2, 1, 1)
        alpha_A, beta_A = Parms_A.chunk(2, dim=1)
        Parms_B = self.down_B(prior_B)
        Parms_B = self.kernel_B(Parms_B.squeeze()).view(-1, C * 2, 1, 1)
        alpha_B, beta_B = Parms_B.chunk(2, dim=1)
        
        q = self.q(_x)
        kv_A = self.kv_A(prior_A)
        k_A,v_A = kv_A.chunk(2, dim=-1)
        kv_B = self.kv_B(prior_B)
        k_B, v_B = kv_B.chunk(2, dim=-1)      

        q = rearrange(q, 'b n (head c) -> b head n c', head=self.num_heads)
        k_A = rearrange(k_A, 'b n (head c) -> b head n c', head=self.num_heads)
        v_A = rearrange(v_A, 'b n (head c) -> b head n c', head=self.num_heads)
        
        k_B = rearrange(k_B, 'b n (head c) -> b head n c', head=self.num_heads)
        v_B = rearrange(v_B, 'b n (head c) -> b head n c', head=self.num_heads)

        attn_A = (q @ k_A.transpose(-2, -1)) * self.scale
        attn_A = attn_A.softmax(dim=-1)

        out_A = (attn_A @ v_A)
        out_A = rearrange(out_A, 'b head n c -> b n (head c)', head=self.num_heads)
        out_A = self.proj_A(out_A)

        
        attn_B = (q @ k_B.transpose(-2, -1)) * self.scale
        attn_B = attn_B.softmax(dim=-1)

        out_B = (attn_B @ v_B)
        out_B = rearrange(out_B, 'b head n c -> b n (head c)', head=self.num_heads)
        out_B = self.proj_B(out_B)
        # sum
        x = x + out_A + out_B
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()
        ## fetaure modulation
        x_A = x * alpha_A + beta_A
        x_B = x * alpha_B + beta_B
        x = x_A + x_B        
        return x


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, group):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, embed_dim, group)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, embed_dim, group)

    def forward(self, x, prior=None):
        x = x + self.attn(self.norm1(x), prior)
        x = x + self.ffn(self.norm2(x), prior)

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x




#########################################################################
##Brightness Adaption Module
class BAGModuleCond(nn.Module):
    """
    Brightness Adaptive Gate (BAG) with PRIOR-driven routing & conditional normalization (FiLM)
    ------------------------------------------------------------------------------------------
    x_g = (1 - w) ⊙ x + w ⊙ x_norm
    where:
      - x_norm uses instance-wise, channel-wise stats, but its affine (gamma, beta) is modulated by a prior.
      - w is produced from a prior of shape (B, P, 128): token-MLP -> (B, P, C), then Linear(P->1) to (B, C).
        Finally w = G(alpha) = alpha^2 / (alpha^2 + eps_gate).

    Args:
        in_channels (int): C of input feature x (B, C, H, W)
        prior_dim (int):   128 (dim of each prior token)
        hidden_prior (int): hidden dim inside token-MLP (per token)
        eps_norm (float):  epsilon for normalization std
        eps_gate (float):  epsilon for G(·) binarization-like squashing
        use_local_stats (bool): if True, use local (window) stats instead of global instance stats
        win (int): window size for local stats (odd)
        film_scale (float): init scale for Δγ, Δβ heads (keep small to stabilize early training)
    """
    def __init__(
        self,
        in_channels: int,
        prior_dim: int = 128,
        hidden_prior: int = 256,
        eps_norm: float = 1e-5,
        eps_gate: float = 1e-6,
        use_local_stats: bool = False,
        win: int = 7,
        film_scale: float = 0.1,
    ):
        super().__init__()
        C = in_channels
        self.C = C
        self.eps_norm = eps_norm
        self.eps_gate = eps_gate
        self.use_local = use_local_stats
        self.win = win

        # Base affine for normalization (will be modulated by FiLM from prior)
        self.gamma_base = nn.Parameter(torch.ones(1, C, 1, 1))
        self.beta_base  = nn.Parameter(torch.zeros(1, C, 1, 1))

        # ---------- PRIOR-DRIVEN ROUTER ----------
        # token-MLP: (B, P, prior_dim) -> (B, P, C)
        self.token_mlp = nn.Sequential(
            nn.Linear(prior_dim, hidden_prior),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_prior, C)  # per-token channel logits
        )
        # Linear(P->1) aggregator for each channel:
        # We'll apply it to tensor shaped (B, C, P): (B, C, P) -> (B, C, 1)
        self.agg_P_to_1 = nn.Linear(in_features=1, out_features=1, bias=False)  # we will use as a trick with view
        # Instead of the trick above, implement a learnable vector a ∈ R^{P} cleanly:
        if C == 32:
            # 通道=32时，聚合向量长度为16
            self.register_parameter('agg_vec', nn.Parameter(torch.randn(1, 1, 16)))
        else:
            # 其它情况，默认长度为4
            self.register_parameter('agg_vec', nn.Parameter(torch.randn(1, 1, 4)))
        self._agg_built = False  # lazy-build agg_vec length = P

        # ---------- FiLM HEADS from PRIOR POOL ----------
        # t = mean-pool over tokens -> (B, prior_dim)
        # FiLM heads produce Δγ, Δβ ∈ ℝ^{B×C}
        self.film_mlp = nn.Sequential(
            nn.Linear(prior_dim, hidden_prior),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_prior, 2 * C)
        )
        # Scale for small init
        self.film_scale = film_scale
        self._init_film_last()

    def _init_film_last(self):
        # make the last linear small so Δγ,Δβ start near 0 (stable)
        last = None
        for m in self.film_mlp.modules():
            if isinstance(m, nn.Linear):
                last = m
        if last is not None:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    @staticmethod
    def _binarize_like_g(alpha: torch.Tensor, eps: float) -> torch.Tensor:
        # G(α) = α^2 / (α^2 + ε)
        alpha2 = alpha * alpha
        return alpha2 / (alpha2 + eps)

    def _stats(self, x: torch.Tensor):
        if not self.use_local:
            mean = x.mean(dim=(2, 3), keepdim=True)
            var  = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        else:
            k = self.win
            pad = k // 2
            mean = F.avg_pool2d(F.pad(x, (pad,pad,pad,pad), mode='reflect'), k, stride=1)
            mean2 = F.avg_pool2d(F.pad(x * x, (pad,pad,pad,pad), mode='reflect'), k, stride=1)
            var = (mean2 - mean * mean).clamp_min(0.0)
        std = torch.sqrt(var + self.eps_norm)
        return mean, std

    def _build_agg_if_needed(self, P: int, device):
        if not self._agg_built or self.agg_vec.shape[-1] != P:
            # learnable vector a ∈ R^{P}, broadcast to (1,1,P)
            self.agg_vec = nn.Parameter(torch.randn(1, 1, P, device=device) / (P ** 0.5))
            self._agg_built = True

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, C, H, W)
            prior: (B, P, 128)  # P tokens of degradation prior
        Returns:
            x_g:   (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.C, f"C mismatch: got {C}, expected {self.C}"
        assert prior.dim() == 3 and prior.size(0) == B, "prior must be (B, P, 128)"

        # ---------- 1) Conditional FiLM for normalization ----------
        # global/token-pooled prior -> Δγ, Δβ
        t = prior.mean(dim=1)                        # (B, 128)
        film = self.film_mlp(t) * self.film_scale   # (B, 2C)
        dgamma, dbeta = film.chunk(2, dim=1)        # (B, C), (B, C)
        dgamma = dgamma.view(B, C, 1, 1)
        dbeta  = dbeta.view(B, C, 1, 1)

        gamma = self.gamma_base * (1.0 + dgamma).clamp(min=0.0)  # keep scale non-negative to preserve monotonicity
        beta  = self.beta_base + dbeta

        # stats
        mean, std = self._stats(x)
        x_hat = (x - mean) / std
        x_norm = gamma * x_hat + beta

        # ---------- 2) PRIOR-driven gate ----------
        Bp, P, D = prior.shape
        self._build_agg_if_needed(P, x.device)

        # token-MLP: (B, P, D) -> (B, P, C)
        h = self.token_mlp(prior)                   # (B, P, C)
        # aggregate over P with a learnable vector a ∈ R^P:
        # (B, P, C) -> (B, C, P) @ (1,1,P)^T -> (B, C, 1)
        h_cp = h.transpose(1, 2)                    # (B, C, P)
        alpha = torch.matmul(h_cp, self.agg_vec.transpose(1,2))  # (B, C, 1)
        alpha = alpha.view(B, C, 1, 1)              # (B, C, 1, 1)

        # binarization-like squashing to [0,1]
        w = self._binarize_like_g(alpha, self.eps_gate)          # (B, C, 1, 1)

        # ---------- 3) Mix ----------
        x_g = (1.0 - w) * x + w * x_norm
        return x_g




##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True), 
                                  nn.Conv2d(n_feat//2, n_feat//2, kernel_size=3, stride=1, padding=1, bias=True)
                                  )

    def forward(self, x):
        return self.body(x)


class BasicLayer_Decoder(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, num_blocks, group, with_contra=False):

        super().__init__()
        self.group = group
        self.with_contra = with_contra
        # build blocks
        ## 这里是Transformer Block的构造 不需要管
        self.blocks = nn.ModuleList([TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group) for i in range(num_blocks)])
        if self.group > 1:
            self.spmm = SPMM(dim, num_heads, bias, embed_dim, LayerNorm_type)
            if with_contra:
                self.contra_att_sem = ContrastAttention(dim, dim)

    def forward(self, x, prior=None):
        # First inject the prior
        if prior is not None and self.group > 1:
            x = self.spmm(x, prior)
            if self.with_contra:
                x = self.contra_att_sem(x)
        prior=None
        ## Then pass through Transformer Blocks
        for blk in self.blocks:
            x = blk(x, prior)
                
        return x
class ContrastAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super(ContrastAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # 计算每个通道的标准差作为对比度
        channel_std = torch.std(x, dim=(2,3))
        
        # 线性映射
        attention_weights = self.fc(channel_std)
        
        # 将权重应用到输入特征图上
        attended_features = torch.einsum('bchw,bc->bchw', x, attention_weights)
        
        return attended_features
        
class BasicLayer_Encoder(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, type_embed_dim, num_blocks, group, with_contra=False, channel=32):

        super().__init__()
        self.group = group
        self.with_contra = with_contra
        # build blocks
        ## 这里是Transformer Block的构造 不需要管
        self.blocks = nn.ModuleList([TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group) for i in range(num_blocks)])
        if self.group > 1:
            self.spmm = SPMM(dim, num_heads, bias, embed_dim, LayerNorm_type)
            self.dpmm = DPMM(dim, num_heads, bias, type_embed_dim, LayerNorm_type, group=group)
            self.cross_att = Cross_Attention(dim, num_heads, bias, LayerNorm_type)
            # print(dim,num_heads,bias,LayerNorm_type)
            if with_contra:
                self.contra_att_sem = ContrastAttention(dim, dim)
                self.contra_att_deg = ContrastAttention(dim, dim)

        self.BA = BAGModuleCond(channel)
        self.featemp = None

    def forward(self, x, prior_sem=None, prior_deg=None):
        # First inject the prior
        if prior_sem is not None and prior_deg is not None and self.group > 1:
            x_sem = self.spmm(x, prior_sem)
            x_deg = self.dpmm(x, prior_deg)
            if self.with_contra:
                x_sem = self.contra_att_sem(x_sem)
                x_deg = self.contra_att_deg(x_deg)
            x = self.cross_att(x_sem, x_deg)
            # print(x_deg.shape, x_sem.shape)
            # print(x.shape)
        prior = None
        ## Then pass through Transformer Blocks
        for blk in self.blocks:
            x = blk(x, prior)
        self.featemp = x
        x = self.BA(x, prior_deg)
        return x
    

class SFP_Header(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, group, task_name='fusion', out_channels=3, num_blocks=4):
        super().__init__()
        self.task_name = task_name
        if task_name in ['vi', 'ir']:
            self.CA = ChannelAttention(dim, ratio=16)            
            self.refinement = BasicLayer_Decoder(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks, group=group)
        else:   
            self.refinement = BasicLayer_Decoder(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks+1, group=group)

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        
    def forward(self, x):
        if self.task_name in ['vi', 'ir']:
            x = self.CA(x) * x
        x = self.refinement(x)
        x = self.output(x)
        # x = self.Tanh(self.output(x))
        return x



class Feature_enhance(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.embed_dims = dim
        self.cross_att_1 = Cross_Attention(32, 2, False, 'WithBias')
        self.cross_att_2 = Cross_Attention(64, 2, False, 'WithBias')
        self.cross_att_3 = Cross_Attention(128, 2, False, 'WithBias')
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        _ofe_cache = os.path.join(_project_root, 'experiments', 'OB_feature', 'features.pth')
        features = torch.load(_ofe_cache, map_location='cpu')
        self.ob_ir_level_1 = features["ir_level_1"]
        self.ob_vi_level_1 = features["vi_level_1"]
        self.ob_ir_level_2 = features["ir_level_2"]
        self.ob_vi_level_2 = features["vi_level_2"]
        self.ob_ir_level_3 = features["ir_level_3"]
        self.ob_vi_level_3 = features["vi_level_3"]

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dims * 2, self.embed_dims * 2 // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims * 2 // reduction, self.embed_dims))
            for _ in range(2)
        ])
        self.sigmoid = nn.Sigmoid()

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 4 * reduction, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(4 * reduction, 1, 1),
                nn.Sigmoid()
            )
            for _ in range(2)
        ])

    def pad_to_divisible(self, x, patch_size=56):
        B, C, W, H = x.shape
        pad_w = (patch_size - W % patch_size) % patch_size
        pad_h = (patch_size - H % patch_size) % patch_size
        # 在右侧和下侧补齐
        x_padded = F.pad(x, (0, pad_h, 0, pad_w), mode='constant', value=0)
        return x_padded, pad_w, pad_h

    def forward(self, feature, patch_size, type):
        feature, pad_w, pad_h = self.pad_to_divisible(feature, patch_size=56)
        B, C, W, H = feature.shape
        n_w = W // patch_size
        n_h = H // patch_size
        N = n_w * n_h
        # 先 reshape 成 [B, C, n_w, 56, n_h, 56]
        feature = feature.reshape(B, C, n_w, patch_size, n_h, patch_size)

        # 调整维度顺序，合并成 [B, C, N, 56, 56]
        feature = feature.permute(0, 1, 2, 4, 3, 5)  # -> [B, C, n_w, n_h, 56, 56]
        feature = feature.reshape(B, C, N, patch_size, patch_size)
        feature_tmp = feature
        feature = feature.permute(0, 2, 1, 3, 4).reshape(-1, C, patch_size, patch_size)

        gap_f = self.avg_pool(feature).view(B*N, self.embed_dims)  ## B*N*N C
        gmp_f = self.max_pool(feature).view(B*N, self.embed_dims)  ## B*N*N C
        ap_f = torch.mean(feature, dim=1, keepdim=True)
        mp_f, _ = torch.max(feature, dim=1, keepdim=True)
        gp_f = torch.cat([gap_f, gmp_f], dim=1)
        p_f = torch.cat([ap_f, mp_f], dim=1)
        gp_f_ca = self.fc[0](gp_f).view(B * N, self.embed_dims, 1, 1)  ## B*N*N C 1 1
        gp_f_ca = self.sigmoid(gp_f_ca)
        p_f_sp = self.mlp[0](p_f)  ## B*N*N 1 R_h R_w

        feature = (feature * gp_f_ca + feature * p_f_sp).view(B, -1, C, patch_size, patch_size).permute((0, 2, 1, 3, 4)) + feature_tmp

        if type == 'ir1':
            refs_flat = self.ob_ir_level_1.view(self.ob_ir_level_1.size(0),-1)
        elif type == 'vi1':
            refs_flat = self.ob_vi_level_1.view(self.ob_vi_level_1.size(0),-1)
        elif type == 'ir2':
            refs_flat = self.ob_ir_level_2.view(self.ob_ir_level_2.size(0),-1)
        elif type == 'vi2':
            refs_flat = self.ob_vi_level_2.view(self.ob_vi_level_2.size(0),-1)
        elif type == 'ir3':
            refs_flat = self.ob_ir_level_3.view(self.ob_ir_level_3.size(0),-1)
        elif type == 'vi3':
            refs_flat = self.ob_vi_level_3.view(self.ob_vi_level_3.size(0),-1)


        feature_flat = feature.view(B, C, N, -1).permute(0, 2, 1, 3).reshape(B, N, -1)
        patches_norm = F.normalize(feature_flat, dim=-1)
        refs_norm = F.normalize(refs_flat, dim=-1)
        sim = torch.matmul(patches_norm, refs_norm.t())
        threshold = 0.1  # 设置阈值
        sim_max, idx = sim.max(dim=-1)  # [B, N]
        mask = (sim_max > threshold)  # [B, N]，True表示要增强的patch
        best_refs = refs_flat[idx]  # [B*N, D]
        best_refs = best_refs.view(B, N, -1)  # [B, N, D]
        best_refs = best_refs * mask.unsqueeze(-1)

        feature_patches = feature_flat.view(B, N, C, patch_size, patch_size)
        best_refs_patches = best_refs.view(B, N, C, patch_size, patch_size)



        # merge batch 和 patch 维度 -> [B*N, C, p, p]
        if type in ['ir1', 'ir2', 'ir3']:
            feature_patches = feature_patches.view(B * N, C, patch_size, patch_size)
            best_refs_patches = best_refs_patches.view(B * N, C, patch_size, patch_size)
            # 判断参考是否“全 0”（给一个小阈值更鲁棒）
            eps = 1e-8
            # 方式1：看最大绝对值是否很小
            valid_mask = (best_refs_patches.abs().amax(dim=(1, 2, 3)) > eps)  # [B*N]，True=有效参考
            # 方式2（可选）：看L1范数
            # valid_mask = (best_refs_patches.abs().sum(dim=(1,2,3)) > eps)

            # 预先用原特征初始化输出，保证梯度与形状一致
            feature_attn = feature_patches.clone()
            f_valid = feature_patches[valid_mask]  # 只取有效样本
            r_valid = best_refs_patches[valid_mask]
            if type == 'ir1':
                v_attn = self.cross_att_1(f_valid, r_valid)
            elif type == 'ir2':
                v_attn = self.cross_att_2(f_valid, r_valid)
            elif type == 'ir3':
                v_attn = self.cross_att_3(f_valid, r_valid)
            feature_attn[valid_mask] = v_attn


        elif type in ['vi1', 'vi2', 'vi3']:
            feature_patches = feature_patches.view(B * N, C, patch_size, patch_size)
            best_refs_patches = best_refs_patches.view(B * N, C, patch_size, patch_size)
            # 判断参考是否“全 0”（给一个小阈值更鲁棒）
            eps = 1e-8
            # 方式1：看最大绝对值是否很小
            valid_mask = (best_refs_patches.abs().amax(dim=(1, 2, 3)) > eps)  # [B*N]，True=有效参考
            # 方式2（可选）：看L1范数
            # valid_mask = (best_refs_patches.abs().sum(dim=(1,2,3)) > eps)

            # 预先用原特征初始化输出，保证梯度与形状一致
            feature_attn = feature_patches.clone()
            if not bool(valid_mask.any()):
                pass

            else:
                f_valid = feature_patches[valid_mask]  # [M, C, p, p]
                r_valid = best_refs_patches[valid_mask]

                Ff = torch.fft.fft2(f_valid)  # complex, [M,C,p,p]
                Fr = torch.fft.fft2(r_valid)

                # 幅度 & 相位
                Af = torch.abs(Ff)  # feature 的幅度
                Pf = torch.angle(Ff)  # feature 的相位
                Ar = torch.abs(Fr)

                mean_Af = Af.mean(dim=(-1, -2), keepdim=True) + eps
                mean_Ar = Ar.mean(dim=(-1, -2), keepdim=True) + eps
                Ar_norm = Ar * (mean_Af / mean_Ar)

                # 用参考幅度 + 原相位重构频谱
                Fnew = Ar_norm * torch.exp(1j * Pf)

                # iFFT 回到空间域，取实部
                f_recon = torch.fft.ifft2(Fnew).real  # [M,C,p,p]
                # 写回只在 valid 的位置
                feature_attn[valid_mask] = f_recon

        feature = 0.2 * feature_attn + 0.8 * feature_patches
        feature = feature.reshape(B, N, C, patch_size, patch_size).permute(0, 2, 1, 3, 4).contiguous()
        feature = feature.reshape(B, C, n_w, n_h, patch_size, patch_size)
        feature = feature.permute(0, 1, 2, 4, 3, 5)
        feature = feature.reshape(B, C, n_w * patch_size, n_h * patch_size)

        orig_W = W - pad_w
        orig_H = H - pad_h
        feature = feature[:, :, :orig_W, :orig_H]

        return feature

##########################################################################
# The implementation builds on Restormer code https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py
@ARCH_REGISTRY.register()
class Transformer_DLR(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        embed_dim = 48,
        type_embed_dim=32, 
        group=4,
        with_contra=False,
        with_SFP=False,
    ):

        super(Transformer_DLR, self).__init__()
         # multi-scale
        self.down_1 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear(group*group, (group*group)//4),
            Rearrange('b c n -> b n c'),
            nn.Linear(embed_dim*4, embed_dim*4)
        )
        self.down_2 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear((group*group)//4, (group*group)//4),
            Rearrange('b c n -> b n c'),
            nn.Linear(embed_dim*4, embed_dim*4)
        )
        self.type_down_1 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear(group*group, (group*group)//4),
            Rearrange('b c n -> b n c'),
            nn.Linear(type_embed_dim*4, type_embed_dim*4)
        )
        
        self.type_down_2 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear((group*group)//4, (group*group)//4),
            Rearrange('b c n -> b n c'),
            nn.Linear(type_embed_dim*4, type_embed_dim*4)
        )
        # self.ir_level_1 = []
        # self.vi_level_1 = []
        # self.ir_level_2 = []
        # self.vi_level_2 = []
        # self.ir_level_3 = []
        # self.vi_level_3 = []

        # self.mask = None

        self.feature_enhance_1 = Feature_enhance(32)
        self.feature_enhance_2 = Feature_enhance(64)
        self.feature_enhance_3 = Feature_enhance(128)

        self.denoise_block1 = DenoiseModule(channels=32, unit_layers=3, act="gelu", hard_mask=True)
        self.denoise_block2 = DenoiseModule(channels=64, unit_layers=3, act="gelu", hard_mask=True)
        self.denoise_block3 = DenoiseModule(channels=128, unit_layers=3, act="gelu", hard_mask=True)
        self.denoise_block4 = DenoiseModule(channels=256, unit_layers=3, act="gelu", hard_mask=True)


        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = BasicLayer_Encoder(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, type_embed_dim=type_embed_dim, num_blocks=num_blocks[0], group=group, with_contra=with_contra, channel=32)
        self.fusion_level1 = Prior_Fusion(dim=dim, num_heads=heads[0], bias=bias, embed_dim=embed_dim, group=group)
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = BasicLayer_Encoder(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, type_embed_dim=type_embed_dim, num_blocks=num_blocks[1], group=group//2, with_contra=with_contra, channel=64)
        self.fusion_level2 = Prior_Fusion(dim=int(dim*2**1), num_heads=heads[1], bias=bias, embed_dim=embed_dim, group=group//2)

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = BasicLayer_Encoder(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, type_embed_dim=type_embed_dim, num_blocks=num_blocks[2], group=group//2, with_contra=with_contra, channel=128)
        self.fusion_level3 = Prior_Fusion(dim=int(dim*2**2), num_heads=heads[2], bias=bias, embed_dim=embed_dim, group=group//2)

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = BasicLayer_Encoder(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, type_embed_dim=type_embed_dim, num_blocks=num_blocks[3], group=group//2, with_contra=with_contra, channel=256)
        self.fusion_latent = Prior_Fusion(dim=int(dim*2**3), num_heads=heads[3], bias=bias, embed_dim=embed_dim, group=group//2)

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = BasicLayer_Decoder(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[2], group=group//2, with_contra=with_contra)


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = BasicLayer_Decoder(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[1], group=group//2, with_contra=with_contra)

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = BasicLayer_Decoder(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[0], group=group, with_contra=with_contra)
        self.with_SFP = with_SFP
        if with_SFP:
            self.f_header = SFP_Header(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group, task_name='fusion', out_channels=out_channels, num_blocks=num_refinement_blocks)
            self.vi_header = SFP_Header(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group, task_name='vi', out_channels=out_channels, num_blocks=num_refinement_blocks)
            self.ir_header = SFP_Header(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group, task_name='ir', out_channels=out_channels, num_blocks=num_refinement_blocks)
        else:
            self.refinement = BasicLayer_Decoder(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_refinement_blocks, group=group, with_contra=with_contra)                
            self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.Tanh = nn.Tanh()


    def forward(self, ir_img, vi_img, deg_prior_A=None, deg_prior_B=None, semantic_prior=None):
        # multi-scale prior 
        img_A = ir_img
        img_B = vi_img

        # multi-scale prior
        prior_1_A = deg_prior_A ##[B, group * group, C]
        prior_2_A = self.type_down_1(prior_1_A) ##[B, group * group // 4, C]
        prior_3_A = self.type_down_2(prior_2_A) ##[B, group * group // 4, C]

        prior_1_B = deg_prior_B
        prior_2_B = self.type_down_1(prior_1_B)
        prior_3_B = self.type_down_2(prior_2_B)
        
        prior_1 = semantic_prior
        prior_2 = self.down_1(prior_1)
        prior_3 = self.down_2(prior_2)

        inp_enc_level1_A = self.patch_embed(img_A)
        out_enc_level1_A = self.encoder_level1(inp_enc_level1_A, prior_1, prior_1_A)
        
        inp_enc_level1_B = self.patch_embed(img_B)
        out_enc_level1_B = self.encoder_level1(inp_enc_level1_B, prior_1, prior_1_B)

        out_enc_level1_A = self.feature_enhance_1(out_enc_level1_A, 56, 'ir1')
        out_enc_level1_B = self.feature_enhance_1(out_enc_level1_B, 56, 'vi1')


        # if out_enc_level1_A.shape[2] == 56 and out_enc_level1_A.shape[3] == 56:
        #     self.ir_level_1.append(out_enc_level1_A)
        # if out_enc_level1_B.shape[2] == 56 and out_enc_level1_B.shape[3] == 56:
        #     self.vi_level_1.append(out_enc_level1_B)

        out_enc_level1_fusion = self.fusion_level1(out_enc_level1_A, out_enc_level1_B, prior_1)
        out_enc_level1_fusion, _, mask = self.denoise_block1(out_enc_level1_fusion)
        # self.mask = mask

        inp_enc_level2_A = self.down1_2(out_enc_level1_A)
        out_enc_level2_A = self.encoder_level2(inp_enc_level2_A, prior_2, prior_2_A)

        inp_enc_level2_B = self.down1_2(out_enc_level1_B)
        out_enc_level2_B = self.encoder_level2(inp_enc_level2_B, prior_2, prior_2_B)

        out_enc_level2_A = self.feature_enhance_2(out_enc_level2_A, 56, 'ir2')
        out_enc_level2_B = self.feature_enhance_2(out_enc_level2_B, 56, 'vi2')

        # if out_enc_level2_A.shape[2] == 56 and out_enc_level2_A.shape[3] == 56:
        #     self.ir_level_2.append(out_enc_level2_A)
        # if out_enc_level2_B.shape[2] == 56 and out_enc_level2_B.shape[3] == 56:
        #     self.vi_level_2.append(out_enc_level2_B)
        
        out_enc_level2_fusion = self.fusion_level2(out_enc_level2_A, out_enc_level2_B, prior_2)
        out_enc_level2_fusion, _, _ = self.denoise_block2(out_enc_level2_fusion)

        inp_enc_level3_A = self.down2_3(out_enc_level2_A)
        out_enc_level3_A = self.encoder_level3(inp_enc_level3_A, prior_3, prior_3_A)         
        
        inp_enc_level3_B = self.down2_3(out_enc_level2_B)
        out_enc_level3_B = self.encoder_level3(inp_enc_level3_B, prior_3, prior_3_B)

        out_enc_level3_A = self.feature_enhance_3(out_enc_level3_A, 56, 'ir3')
        out_enc_level3_B = self.feature_enhance_3(out_enc_level3_B, 56, 'vi3')

        # if out_enc_level3_A.shape[2] == 56 and out_enc_level3_A.shape[3] == 56:
        #     self.ir_level_3.append(out_enc_level3_A)
        # if out_enc_level3_B.shape[2] == 56 and out_enc_level3_B.shape[3] == 56:
        #     self.vi_level_3.append(out_enc_level3_B)
        
        out_enc_level3_fusion = self.fusion_level3(out_enc_level3_A, out_enc_level3_B, prior_3)
        out_enc_level3_fusion, _, _ = self.denoise_block3(out_enc_level3_fusion)

        inp_enc_level4_A = self.down3_4(out_enc_level3_A)        
        latent_A = self.latent(inp_enc_level4_A, prior_3, prior_3_A) 
        
        inp_enc_level4_B = self.down3_4(out_enc_level3_B) 
        latent_B = self.latent(inp_enc_level4_B, prior_3, prior_3_B)

        
        latent_fusion = self.fusion_latent(latent_A, latent_B, prior_3)
        latent_fusion, _, _ = self.denoise_block4(latent_fusion)
                        
        inp_dec_level3 = self.up4_3(latent_fusion)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3_fusion], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3, prior_3) ##[B, group * group // 4, C]

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2_fusion], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2, prior_2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1_fusion], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1, prior_1)
        if self.with_SFP:
            vi_dec = self.vi_header(out_dec_level1) + img_B 
            ir_dec = self.ir_header(out_dec_level1) + img_A 
            f_dec = self.f_header(out_dec_level1) + img_B 
             # out_dec_level1 = out_dec_level1.clamp(0, 1)
            f_img = (self.Tanh(f_dec) + 1) / 2
            ir_img = (self.Tanh(ir_dec) + 1) / 2
            vi_img = (self.Tanh(vi_dec) + 1) / 2
            # out_dec_level1 = (out_dec_level1 - torch.min(out_dec_level1)) / (torch.max(out_dec_level1) - torch.min(out_dec_level1))
            results = {'fusion':f_img, 'ir':ir_img, 'vi':vi_img}
            return results
        else:
            out_dec_level1 = self.refinement(out_dec_level1, prior_1)    
            f_dec = self.output(out_dec_level1) + img_B ## 是否跳接？ 尝试一下不跳接的效果            
            f_img = (self.Tanh(f_dec) + 1) / 2
            results = {'fusion':f_img, 'ir':f_img, 'vi':f_img}
            return results


@ARCH_REGISTRY.register()
class Transformer_gen(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 embed_dim=48,
                 type_embed_dim=32,
                 group=4,
                 with_contra=False,
                 with_SFP=False,
                 ):

        super(Transformer_gen, self).__init__()
        # multi-scale
        self.down_1 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear(group * group, (group * group) // 4),
            Rearrange('b c n -> b n c'),
            nn.Linear(embed_dim * 4, embed_dim * 4)
        )
        self.down_2 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear((group * group) // 4, (group * group) // 4),
            Rearrange('b c n -> b n c'),
            nn.Linear(embed_dim * 4, embed_dim * 4)
        )
        self.type_down_1 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear(group * group, (group * group) // 4),
            Rearrange('b c n -> b n c'),
            nn.Linear(type_embed_dim * 4, type_embed_dim * 4)
        )

        self.type_down_2 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear((group * group) // 4, (group * group) // 4),
            Rearrange('b c n -> b n c'),
            nn.Linear(type_embed_dim * 4, type_embed_dim * 4)
        )


        self.feature_enhance_1 = Feature_enhance(32)
        self.feature_enhance_2 = Feature_enhance(64)
        self.feature_enhance_3 = Feature_enhance(128)

        self.denoise_block1 = DenoiseModule(channels=32, unit_layers=3, act="gelu", hard_mask=True)
        self.denoise_block2 = DenoiseModule(channels=64, unit_layers=3, act="gelu", hard_mask=True)
        self.denoise_block3 = DenoiseModule(channels=128, unit_layers=3, act="gelu", hard_mask=True)
        self.denoise_block4 = DenoiseModule(channels=256, unit_layers=3, act="gelu", hard_mask=True)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = BasicLayer_Encoder(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                                 bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim,
                                                 type_embed_dim=type_embed_dim, num_blocks=num_blocks[0], group=group,
                                                 with_contra=with_contra, channel=32)


        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = BasicLayer_Encoder(dim=int(dim * 2 ** 1), num_heads=heads[1],
                                                 ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                 LayerNorm_type=LayerNorm_type, embed_dim=embed_dim,
                                                 type_embed_dim=type_embed_dim, num_blocks=num_blocks[1],
                                                 group=group // 2, with_contra=with_contra, channel=64)


        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = BasicLayer_Encoder(dim=int(dim * 2 ** 2), num_heads=heads[2],
                                                 ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                 LayerNorm_type=LayerNorm_type, embed_dim=embed_dim,
                                                 type_embed_dim=type_embed_dim, num_blocks=num_blocks[2],
                                                 group=group // 2, with_contra=with_contra, channel=128)


        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = BasicLayer_Encoder(dim=int(dim * 2 ** 3), num_heads=heads[3],
                                         ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                         LayerNorm_type=LayerNorm_type, embed_dim=embed_dim,
                                         type_embed_dim=type_embed_dim, num_blocks=num_blocks[3], group=group // 2,
                                         with_contra=with_contra, channel=256)


        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = BasicLayer_Decoder(dim=int(dim * 2 ** 2), num_heads=heads[2],
                                                 ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                 LayerNorm_type=LayerNorm_type, embed_dim=embed_dim,
                                                 num_blocks=num_blocks[2], group=group // 2, with_contra=with_contra)

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = BasicLayer_Decoder(dim=int(dim * 2 ** 1), num_heads=heads[1],
                                                 ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                 LayerNorm_type=LayerNorm_type, embed_dim=embed_dim,
                                                 num_blocks=num_blocks[1], group=group // 2, with_contra=with_contra)

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = BasicLayer_Decoder(dim=int(dim * 2 ** 1), num_heads=heads[0],
                                                 ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                 LayerNorm_type=LayerNorm_type, embed_dim=embed_dim,
                                                 num_blocks=num_blocks[0], group=group, with_contra=with_contra)


        self.vi_header = SFP_Header(dim=int(dim * 2 ** 1), num_heads=heads[0],
                                    ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                    LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group, task_name='vi',
                                    out_channels=out_channels, num_blocks=num_refinement_blocks)
        self.ir_header = SFP_Header(dim=int(dim * 2 ** 1), num_heads=heads[0],
                                    ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                    LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group, task_name='ir',
                                    out_channels=out_channels, num_blocks=num_refinement_blocks)
        self.Tanh = nn.Tanh()


    def forward(self, ir_img, vi_img, deg_prior_A=None, deg_prior_B=None, semantic_prior=None):
        # multi-scale prior
        img_A = ir_img
        img_B = vi_img

        # multi-scale prior
        prior_1_A = deg_prior_A  ##[B, group * group, C]
        prior_2_A = self.type_down_1(prior_1_A)  ##[B, group * group // 4, C]
        prior_3_A = self.type_down_2(prior_2_A)  ##[B, group * group // 4, C]

        prior_1_B = deg_prior_B
        prior_2_B = self.type_down_1(prior_1_B)
        prior_3_B = self.type_down_2(prior_2_B)

        prior_1 = semantic_prior
        prior_2 = self.down_1(prior_1)
        prior_3 = self.down_2(prior_2)

        inp_enc_level1_A = self.patch_embed(img_A)
        out_enc_level1_A = self.encoder_level1(inp_enc_level1_A, prior_1, prior_1_A)

        inp_enc_level1_B = self.patch_embed(img_B)
        out_enc_level1_B = self.encoder_level1(inp_enc_level1_B, prior_1, prior_1_B)

        out_enc_level1_A = self.feature_enhance_1(out_enc_level1_A, 56, 'ir1')
        out_enc_level1_B = self.feature_enhance_1(out_enc_level1_B, 56, 'vi1')

        out_enc_level1_A, _, _ = self.denoise_block1(out_enc_level1_A)
        out_enc_level1_B, _, _ = self.denoise_block1(out_enc_level1_B)


        inp_enc_level2_A = self.down1_2(out_enc_level1_A)
        out_enc_level2_A = self.encoder_level2(inp_enc_level2_A, prior_2, prior_2_A)

        inp_enc_level2_B = self.down1_2(out_enc_level1_B)
        out_enc_level2_B = self.encoder_level2(inp_enc_level2_B, prior_2, prior_2_B)

        out_enc_level2_A = self.feature_enhance_2(out_enc_level2_A, 56, 'ir2')
        out_enc_level2_B = self.feature_enhance_2(out_enc_level2_B, 56, 'vi2')

        out_enc_level2_A, _, _ = self.denoise_block2(out_enc_level2_A)
        out_enc_level2_B, _, _ = self.denoise_block2(out_enc_level2_B)

        inp_enc_level3_A = self.down2_3(out_enc_level2_A)
        out_enc_level3_A = self.encoder_level3(inp_enc_level3_A, prior_3, prior_3_A)

        inp_enc_level3_B = self.down2_3(out_enc_level2_B)
        out_enc_level3_B = self.encoder_level3(inp_enc_level3_B, prior_3, prior_3_B)

        out_enc_level3_A = self.feature_enhance_3(out_enc_level3_A, 56, 'ir3')
        out_enc_level3_B = self.feature_enhance_3(out_enc_level3_B, 56, 'vi3')

        out_enc_level3_A, _, _ = self.denoise_block3(out_enc_level3_A)
        out_enc_level3_B, _, _ = self.denoise_block3(out_enc_level3_B)


        inp_enc_level4_A = self.down3_4(out_enc_level3_A)
        latent_A = self.latent(inp_enc_level4_A, prior_3, prior_3_A)

        inp_enc_level4_B = self.down3_4(out_enc_level3_B)
        latent_B = self.latent(inp_enc_level4_B, prior_3, prior_3_B)

        latent_A, _, _ = self.denoise_block4(latent_A)
        latent_B, _, _ = self.denoise_block4(latent_B)

        inp_dec_level3A = self.up4_3(latent_A)
        inp_dec_level3A = torch.cat([inp_dec_level3A, out_enc_level3_A], 1)
        inp_dec_level3A = self.reduce_chan_level3(inp_dec_level3A)
        out_dec_level3A = self.decoder_level3(inp_dec_level3A, prior_3)  ##[B, group * group // 4, C]

        inp_dec_level3B = self.up4_3(latent_B)
        inp_dec_level3B = torch.cat([inp_dec_level3B, out_enc_level3_B], 1)
        inp_dec_level3B = self.reduce_chan_level3(inp_dec_level3B)
        out_dec_level3B = self.decoder_level3(inp_dec_level3B, prior_3)  ##[B, group * group // 4, C]

        inp_dec_level2A = self.up3_2(out_dec_level3A)
        inp_dec_level2A = torch.cat([inp_dec_level2A, out_enc_level2_A], 1)
        inp_dec_level2A = self.reduce_chan_level2(inp_dec_level2A)
        out_dec_level2A = self.decoder_level2(inp_dec_level2A, prior_2)

        inp_dec_level2B = self.up3_2(out_dec_level3B)
        inp_dec_level2B = torch.cat([inp_dec_level2B, out_enc_level2_B], 1)
        inp_dec_level2B = self.reduce_chan_level2(inp_dec_level2B)
        out_dec_level2B = self.decoder_level2(inp_dec_level2B, prior_2)

        inp_dec_level1A = self.up2_1(out_dec_level2A)
        inp_dec_level1A = torch.cat([inp_dec_level1A, out_enc_level1_A], 1)
        out_dec_level1A = self.decoder_level1(inp_dec_level1A, prior_1)

        inp_dec_level1B = self.up2_1(out_dec_level2B)
        inp_dec_level1B = torch.cat([inp_dec_level1B, out_enc_level1_B], 1)
        out_dec_level1B = self.decoder_level1(inp_dec_level1B, prior_1)

        vi_dec = self.vi_header(out_dec_level1B) + img_B
        ir_dec = self.ir_header(out_dec_level1A) + img_A

        ir_img = (self.Tanh(ir_dec) + 1) / 2
        vi_img = (self.Tanh(vi_dec) + 1) / 2
        # out_dec_level1 = (out_dec_level1 - torch.min(out_dec_level1)) / (torch.max(out_dec_level1) - torch.min(out_dec_level1))
        results = {'ir': ir_img, 'vi': vi_img}
        return results

