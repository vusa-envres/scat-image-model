import math
import numpy as np
#import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class PatchEmbed(nn.Module):

    def __init__(self, img_size=[24,96], stem_conv=False, patch_size=8, hidden_dim=64, embed_dim=192):
        super().__init__()
        assert patch_size % 2 == 0
        assert img_size[0] % patch_size == 0
        assert img_size[1] % patch_size == 0
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.proj = nn.Conv2d(hidden_dim, embed_dim,
                              kernel_size=patch_size // 2,
                              stride=patch_size // 2)
        #self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


class OutlookAttention(nn.Module):

    def __init__(self, embed_size, num_heads, kernel_size=3, padding=1, stride=1,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert embed_size % num_heads == 0
        
        head_dim = embed_size // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = qk_scale or head_dim**-0.5
        self.v = nn.Linear(embed_size, embed_size, bias=qkv_bias)
        self.attn = nn.Linear(embed_size, kernel_size**4 * num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_size, embed_size)
        self.proj_drop = nn.Dropout(proj_drop)
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):

        B, H, W, C = x.shape
        c = C // self.num_heads
        h = math.ceil(H / self.stride)
        w = math.ceil(W / self.stride)
        kxk = self.kernel_size * self.kernel_size

        v = self.v(x)
        v = v.permute(0, 3, 1, 2)  # B, C, H, W
        v = self.unfold(v).reshape(B, self.num_heads, c, kxk, h * w)
        v = v.permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H

        x = x.permute(0, 3, 1, 2)
        attn = self.pool(x)
        attn = attn.permute(0, 2, 3, 1)
        attn = self.attn(attn)
        attn = attn.reshape(B, h*w, self.num_heads, kxk, kxk)
        attn = attn.permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.permute(0, 1, 4, 3, 2)
        x = x.reshape(B, C * kxk, h * w)
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size,
                   padding=self.padding, stride=self.stride)
        x = x.permute(0, 2, 3, 1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Outlooker(nn.Module):

    def __init__(self, dim, kernel_size, padding, stride=1,
                 num_heads=1, mlp_ratio=3., 
                 attn_drop=0., drop_path=0., 
                 qkv_bias=False, qk_scale=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = OutlookAttention(dim, num_heads, kernel_size=kernel_size,
                                     padding=padding, stride=stride,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     attn_drop=attn_drop)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x1 = self.norm1(x)
        x1 = self.attn(x1)
        x += x1
        x1 = self.norm2(x)
        x1 = self.mlp(x1)
        x += x1
        return x


class Attention(nn.Module):
    def __init__(self, embed_size,  num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert embed_size % num_heads == 0
        self.num_heads = num_heads
        head_dim = embed_size // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(embed_size, embed_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_size, embed_size)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        c = C // self.num_heads

        qvk = self.qkv(x)
        qvk = qvk.reshape(B, H * W, 3, self.num_heads, c)
        qvk = qvk.permute(2, 0, 3, 1, 4)
        q, k, v = qvk[0], qvk[1], qvk[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Transformer(nn.Module):
    def __init__(self, embed_size, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.attn = Attention(embed_size, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop)
        self.norm2 = nn.LayerNorm(embed_size)
        mlp_hidden_dim = int(embed_size * mlp_ratio)
        self.mlp = Mlp(in_features=embed_size, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x1 = self.norm1(x)
        x1 = self.attn(x1)
        x += x1
        x1 = self.norm2(x)
        x1 = self.mlp(x1)
        x += x1
        return x


class ClassAttention(nn.Module):
    def __init__(self, embed_size, num_heads=8, head_dim=None, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            head_dim = embed_size // num_heads
            self.head_dim = head_dim
        self.scale = qk_scale or head_dim**-0.5

        self.kv = nn.Linear(embed_size, self.head_dim * self.num_heads * 2, bias=qkv_bias)
        self.q = nn.Linear(embed_size, self.head_dim * self.num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, embed_size)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        kv = self.kv(x)
        kv = kv.reshape(B, N, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        q = self.q(x[:, :1, :])
        q = q.reshape(B, self.num_heads, 1, self.head_dim)
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        cls_embed = (attn @ v)
        cls_embed = cls_embed.transpose(1, 2)
        cls_embed = cls_embed.reshape(B, 1, self.head_dim * self.num_heads)
        cls_embed = self.proj(cls_embed)
        cls_embed = self.proj_drop(cls_embed)
        return cls_embed


class ClassBlock(nn.Module):
    def __init__(self, embed_size, num_heads, head_dim=None, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.attn = ClassAttention(embed_size, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(embed_size)
        mlp_hidden_dim = int(embed_size * mlp_ratio)
        self.mlp = Mlp(in_features=embed_size, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        cls_embed = x[:, :1]
        x1 = self.norm1(x)
        x1 = self.attn(x1)
        cls_embed += x1
        x1 = self.norm2(cls_embed)
        x1 = self.mlp(x1)
        cls_embed += x1
        return torch.cat([cls_embed, x[:, 1:]], dim=1)


class VOLO(nn.Module):

    def __init__(self, layers, img_size=[24,96], num_classes=16, patch_size=8,
                 stem_hidden_dim=64, embed_dims=[192, 384, 384, 384], 
                 num_heads=[3, 6, 6, 6], downsamples=[True, False, False, False],
                 outlook_attention=[True, False, False, False], 
                 mlp_ratios=[1, 1, 1, 1], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 num_post_layers=2, return_dense=True, mix_token=True,
                 pooling_scale=2, out_kernel=3, out_stride=2, out_padding=1):

        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(stem_conv=True, patch_size=patch_size,
                                      hidden_dim=stem_hidden_dim,
                                      embed_dim=embed_dims[0])

        # inital positional encoding, we add positional encoding after outlooker blocks
        self.pos_embed = nn.Parameter(torch.zeros(1, img_size[0] // patch_size // pooling_scale,
                        img_size[1] // patch_size // pooling_scale, embed_dims[-1]))

        self.pos_drop = nn.Dropout(p=drop_rate)

        network = []
        # Outlooker layers
        blocks = []
        for block_idx in range(layers[0]):
            block_dpr = drop_path_rate * block_idx / (sum(layers) - 1)
            blocks.append(Outlooker(embed_dims[0], kernel_size=out_kernel, padding=out_padding,
                                   stride=out_stride, num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                   qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate))
        blocks = nn.Sequential(*blocks)
        network.append(blocks)
        
        # Downsample layers
        network.append(Downsample(embed_dims[0], embed_dims[1], 2))
        
        # Outlooker layers
        for i in range(1, len(layers)):
            blocks = []
            for block_idx in range(layers[i]):
                block_dpr = drop_path_rate * (block_idx + sum(layers[:i])) / (sum(layers) - 1)
                blocks.append(Transformer(embed_dims[i], num_heads[i],
                             mlp_ratio=mlp_ratios[i],
                             qkv_bias=qkv_bias,
                             qk_scale=qk_scale,
                             attn_drop=attn_drop_rate))
            blocks = nn.Sequential(*blocks)
            network.append(blocks)

        self.network = nn.ModuleList(network)

        self.post_network = []
        for i in range(num_post_layers):
            self.post_network.append(
                ClassBlock( embed_size=embed_dims[-1],
                          num_heads=num_heads[-1],
                          mlp_ratio=mlp_ratios[-1],
                          qkv_bias=qkv_bias,
                          qk_scale=qk_scale,
                          attn_drop=attn_drop_rate))
        self.post_network = nn.Sequential(*(self.post_network) )
        self.post_network = nn.ModuleList(self.post_network)
            
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1]))
        trunc_normal_(self.cls_token, std=.02)

        # set output type
        #self.return_mean = return_mean  # if yes, return mean, not use class token
        self.return_dense = return_dense  # if yes, return class token and all feature tokens
        #if return_dense:
        #    assert not return_mean, "cannot return both mean and dense"
        self.mix_token = mix_token
        self.pooling_scale = pooling_scale
        if mix_token:  # enable token mixing, see token labeling for details.
            self.beta = 1.0
            assert return_dense, "return all tokens if mix_token is enabled"
        if return_dense:
            self.aux_head = nn.Linear(
                embed_dims[-1],
                num_classes) if num_classes > 0 else nn.Identity()
        self.norm = nn.LayerNorm(embed_dims[-1])

        # Classifier head
        self.head = nn.Linear(embed_dims[-1], num_classes)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        
        # step1: patch embedding
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)       # B,C,H,W-> B,H,W,C
        bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0

        for idx, block in enumerate(self.network):
            if idx == 2:  # add positional encoding after outlooker blocks
                x = x + self.pos_embed
                x = self.pos_drop(x)
            x = block(x)

        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)

        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.post_network:
            x = block(x)

        x = self.norm(x)

        x_cls = self.head(x[:, 0])

        x_aux = self.aux_head(x[:, 1:])

        if not self.training:
            return x_cls + 0.5 * x_aux.max(1)[0]

        return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)


