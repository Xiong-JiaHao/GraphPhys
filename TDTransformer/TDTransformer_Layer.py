"""
Adapted from https://github.com/lukemelas/simple-bert
"""

import numpy as np
from torch.nn import functional as F
import torch
from torch import nn
import math
import os

'''
Temporal Center-difference based Convolutional layer (3D version)
theta: control the percentage of original convolution and centeral-difference convolution
'''


class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention_TDC_gra_sharp(nn.Module):
    """Multi-Headed Dot Product Attention with depth-wise Conv3d"""

    def __init__(self, dim, num_heads, dropout, theta):
        super().__init__()

        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
            # nn.ELU(),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
            # nn.ELU(),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
            # nn.BatchNorm3d(dim),
            # nn.ELU(),
        )

        # self.proj_q = nn.Linear(dim, dim)
        # self.proj_k = nn.Linear(dim, dim)
        # self.proj_v = nn.Linear(dim, dim)

        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None  # for visualization

    def forward(self, x, gra_sharp):  # [B, 4*4*40, 128]
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        [B, P, C] = x.shape
        x = x.transpose(1, 2).view(B, C, P // 16, 4, 4)  # [B, dim, 40, 4, 4]
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q = q.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        k = k.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        v = v.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]

        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / gra_sharp

        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h, scores


class PositionWiseFeedForward_ST(nn.Module):
    """FeedForward Neural Networks for each position"""

    def __init__(self, dim, ff_dim):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Conv3d(dim, ff_dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )

        self.STConv = nn.Sequential(
            nn.Conv3d(ff_dim, ff_dim, 3, stride=1, padding=1, groups=ff_dim, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )

        self.fc2 = nn.Sequential(
            nn.Conv3d(ff_dim, dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(dim),
        )

    def forward(self, x):  # [B, 4*4*40, 128]
        [B, P, C] = x.shape
        # x = x.transpose(1, 2).view(B, C, 40, 4, 4)      # [B, dim, 40, 4, 4]
        x = x.transpose(1, 2).view(B, C, P // 16, 4, 4)  # [B, dim, 40, 4, 4]
        x = self.fc1(x)  # x [B, ff_dim, 40, 4, 4]
        x = self.STConv(x)  # x [B, ff_dim, 40, 4, 4]
        x = self.fc2(x)  # x [B, dim, 40, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]

        return x

        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        # return self.fc2(F.gelu(self.fc1(x)))


class Block_ST_TDC_gra_sharp(nn.Module):
    """Transformer Block"""

    def __init__(self, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.attn = MultiHeadedSelfAttention_TDC_gra_sharp(dim, num_heads, dropout, theta)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward_ST(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gra_sharp):
        Atten, Score = self.attn(self.norm1(x), gra_sharp)
        h = self.drop(self.proj(Atten))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x, Score


class Transformer_ST_TDC_gra_sharp(nn.Module):
    """Transformer with Self-Attentive Blocks"""

    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block_ST_TDC_gra_sharp(dim, num_heads, ff_dim, dropout, theta) for _ in range(num_layers)])

    def forward(self, x, gra_sharp):
        for block in self.blocks:
            x, Score = block(x, gra_sharp)
        return x, Score


class TDTransformer(nn.Module):
    def __init__(self, opt):
        super(TDTransformer, self).__init__()
        patches = opt.tdt_patches
        dim = opt.tdt_dim
        ff_dim = opt.tdt_ff_dim
        num_heads = opt.tdt_num_heads
        num_layers = opt.tdt_num_layers
        dropout_rate = opt.tdt_dropout_rate
        theta = opt.tdt_theta
        self.gra_sharp = opt.tdt_gra_sharp
        self.image_size = opt.tdt_image_size
        self.dim = dim

        # Image and patch sizes
        t, h, w = as_tuple(self.image_size)  # tube sizes
        ft, fh, fw = as_tuple(patches)  # patch sizes, ft = 4 ==> 160/4=40
        gt, gh, gw = t // ft, h // fh, w // fw  # number of patches
        seq_len = gh * gw * gt

        # Patch embedding    [4x16x16]conv
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))

        # Transformer
        self.transformer1 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
                                                         ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        # Transformer
        self.transformer2 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
                                                         ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        # Transformer
        self.transformer3 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
                                                         ff_dim=ff_dim, dropout=dropout_rate, theta=theta)

        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim // 4, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(dim // 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        # self.normLast = nn.LayerNorm(dim, eps=1e-6)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim // 2, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim // 2),
            nn.ELU(),
        )

        # self.ConvBlockLast = nn.Conv1d(dim // 2, 1, 1, stride=1, padding=0)

    def forward(self, x):
        # x = torch.stack(x, dim=2)  # [B, 96, 160, 16, 16]
        # b, _, t, _, _ = x.shape  # [B, 96, 160, 16, 16]

        b, c, t, fh, fw = x.shape

        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)  # [B, 64, 160, 64, 64]

        x = self.patch_embedding(x)  # [B, 96, 40, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [B, 40*4*4, 96]

        Trans_features, Score1 = self.transformer1(x, self.gra_sharp)  # [B, 4*4*40, 96]
        Trans_features2, Score2 = self.transformer2(Trans_features, self.gra_sharp)  # [B, 4*4*40, 96]
        Trans_features3, Score3 = self.transformer3(Trans_features2, self.gra_sharp)  # [B, 4*4*40, 96]

        # Trans_features3 = self.normLast(Trans_features3)

        # upsampling heads
        # features_last = Trans_features3.transpose(1, 2).view(b, self.dim, 40, 4, 4) # [B, 96, 40, 4, 4]
        features_last = Trans_features3.transpose(1, 2).view(b, self.dim, t // 4, 4, 4)  # [B, 96, 40, 4, 4]

        features_last = self.upsample(features_last)  # x [B, 96, 80, 4, 4]
        features_last = self.upsample2(features_last)  # x [B, 48, 160, 4, 4]

        # features_last = torch.mean(features_last, 3)  # x [B, 48, 160, 4]
        # features_last = torch.mean(features_last, 3)  # x [B, 48, 160]
        # rPPG = self.ConvBlockLast(features_last)  # x [B, 1, 160]
        # rPPG = rPPG.squeeze(1)  # x [B, 160]
        return features_last


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


# 将 TDTransformer 抽出来
# 配置文件详见论文 https://arxiv.org/pdf/2111.12082v1.pdf
def td_transformer(opt):
    model = TDTransformer(opt)
    path = os.path.dirname(__file__) + '/Physformer_VIPL_fold1.pkl'
    model.load_state_dict(torch.load(path))
    return model
