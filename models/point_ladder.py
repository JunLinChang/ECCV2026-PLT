import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_
import numpy as np

from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from knn_cuda import KNN, knn
# from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation, QueryAndGroup, grouping_operation
from einops import rearrange
from einops.layers.torch import Rearrange
# from extensions.pointops.functions import pointops


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx
    

class SABlock_KNN(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, k=16, r=8, use_res=True, use_act=True, dropout=0.):
        super().__init__()
        self.stride = stride
        self.k = k
        self.use_res = use_res
        self.use_act = use_act

        self.convs = nn.Sequential(
            nn.Conv2d(in_dim + 3, out_dim, 1)
        )
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.GELU()
        
        if use_res:
            self.res_conv = nn.Sequential(
                nn.Linear(in_dim, out_dim)
            )
            # self.dropout = DropPath(dropout) if dropout > 0. else nn.Identity()
            self.dropout = nn.Dropout(dropout)
        
        self.apply(self.__init_weight)
        
    def __init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, xyz):
        # x: (b, n, d)
        # xyz: (b, n, 3)
        # out: (b, n // stride, d')
        fps_idx = furthest_point_sample(xyz, x.size(1) // self.stride) # (b, n // stride)
        new_xyz = gather_operation(xyz.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous() # (b, n // stride, 3)
        new_x = gather_operation(x.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous() # (b, n // stride, d)
        neighborhood_idx = knn_point(self.k, xyz, new_xyz).int()
        neighborhood_xyz = grouping_operation(xyz.transpose(1, 2).contiguous(), neighborhood_idx) # (b, 3, n // stride, k)
        neighborhood_x = grouping_operation(x.transpose(1, 2).contiguous(), neighborhood_idx) # (b, d, n // stride, k)
        neighborhood_x = torch.cat([neighborhood_x, neighborhood_xyz - new_xyz.transpose(1, 2).unsqueeze(-1)], dim=1) # (b, d + 3, n // stride, k)
        # print(neighborhood_x.size())
        x = self.convs(neighborhood_x)
        x = self.norm(x)
        if self.use_res:
            x = x.max(dim=-1)[0].transpose(1, 2)    # (b, n // stride, d')
            x = self.res_conv(new_x) + self.dropout(x)    # (b, n // stride, d')
            x = self.act(x)
        else:
            x = self.act(x)
            x = x.max(dim=-1)[0].transpose(1, 2)    # (b, n // stride, d')
        return x, new_xyz
    

class LocalAggregation_KNN(nn.Module):
    def __init__(self, in_dim, out_dim, k=16, r=8):
        super().__init__()
        self.k = k
        
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim + 3, out_dim, 1)
        )
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.GELU()
        
        self.apply(self.__init_weight)
        
    def __init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x1, xyz1, x2, xyz2):
        # Key:
        # x1: (b, n, d)
        # xyz1: (b, n, 3)
        # Query:
        # x2: (b, m, d)
        # xyz2: (b, m, 3)
        # out: (b, n, d)
        neighborhood_idx = knn_point(self.k, xyz1, xyz2).int()
        neighborhood_xyz = grouping_operation(xyz1.transpose(1, 2).contiguous(), neighborhood_idx) # (b, 3, n, k)
        neighborhood_x = grouping_operation(x1.transpose(1, 2).contiguous(), neighborhood_idx) # (b, d, n, k)
        neighborhood_x = torch.cat([neighborhood_x, neighborhood_xyz - xyz2.transpose(1, 2).unsqueeze(-1)], dim=1) # (b, d + 3, n, k)
        x = self.convs(neighborhood_x)
        x = self.norm(x)
        x = self.act(x)
        x = x.max(dim=-1)[0].transpose(1, 2)    # (b, n, d)
        return x
    

class Adapter_KNN(nn.Module):
    def __init__(self, dim, transformer_dim, k=16, r=8, r_ratio=8, dropout=0.):
        super().__init__()
        self.k = k

        self.local_aggregation1 = LocalAggregation_KNN(dim, dim, k, r)
        self.local_aggregation2 = LocalAggregation_KNN(dim, dim, 16, r)

        hidden_dim = dim // r_ratio if dim // r_ratio > 16 else 16
        
        self.linear = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim * 2)
        ) 

        self.proj = nn.Sequential(
            nn.Linear(dim, dim)
        )

        self.act = nn.GELU()

        self.dropout = nn.Dropout(dropout)
        # self.dropout = DropPath(dropout) if dropout > 0. else nn.Identity()
        
        self.apply(self.__init_weight)
        
    def __init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x1, xyz1, x2, xyz2):
        x_self = self.local_aggregation1(x2, xyz2, x2, xyz2)
        x_cross = self.local_aggregation2(x1, xyz1, x2, xyz2)
        
        x_self_global = x_self.mean(dim=1)
        x_cross_global = x_cross.mean(dim=1)
        
        x_alpha = x_self_global + x_cross_global
        x_alpha = self.linear(x_alpha)
        x_alpha = rearrange(x_alpha, 'B (r C)->r B C', r=2)
        x_alpha = F.softmax(x_alpha, dim=0)
        
        x_self = x_self * x_alpha[0].unsqueeze(1)
        x_cross = x_cross * x_alpha[1].unsqueeze(1)
        
        x = self.dropout(self.proj(x_self + x_cross)) + x2
        x = self.act(x)
        
        return x


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.tfts_gamma_1, self.tfts_beta_1 = init_tfts(hidden_features)
        self.tfts_gamma_2, self.tfts_beta_2 = init_tfts(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = apply_tfts(x, self.tfts_gamma_1, self.tfts_beta_1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = apply_tfts(x, self.tfts_gamma_2, self.tfts_beta_2)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.tfts_gamma_1, self.tfts_beta_1 = init_tfts(dim * 3)
        self.tfts_gamma_2, self.tfts_beta_2 = init_tfts(dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = apply_tfts(self.qkv(x), self.tfts_gamma_1,
                         self.tfts_beta_1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = apply_tfts(x, self.tfts_gamma_2, self.tfts_beta_2)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.tfts_gamma_1, self.tfts_beta_1 = init_tfts(dim)
        self.tfts_gamma_2, self.tfts_beta_2 = init_tfts(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(apply_tfts(self.norm1(x), self.tfts_gamma_1, self.tfts_beta_1)))
        x = x + self.drop_path(self.mlp(apply_tfts(self.norm2(x), self.tfts_gamma_2, self.tfts_beta_2)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., adapter_dim=[16, 32, 64], stride=4, r=8, k=16, radius=0.1, r_ratio=8, n_blocks=[1, 1, 1], dropout=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        
        self.time_step = depth // (len(adapter_dim) - 1)
        self.idx = np.arange(depth - 1, -1, -self.time_step)[::-1]
        # print(f'Index: {self.idx}')
        
        dropout = dropout if isinstance(dropout, list) else torch.linspace(0, dropout, len(adapter_dim))
        print(f'Dropout: {dropout}')
        for i in range(len(adapter_dim) - 1):
            # print(f'Adapter {i}: {adapter_dim[i]} -> {adapter_dim[i + 1]}')
            setattr(self, f'adapter_proj{i}', nn.Linear(embed_dim, adapter_dim[i + 1], bias=False))
            setattr(self, f'adapter_down{i}', SABlock_KNN(adapter_dim[i], adapter_dim[i + 1], stride=stride[i + 1], r=r[i + 1], k=k[i + 1], dropout=dropout[i + 1]))
            setattr(self, f'adapter{i}', nn.ModuleList([Adapter_KNN(adapter_dim[i + 1], embed_dim, k=k[i + 1], r=r[i + 1], r_ratio=r_ratio, dropout=dropout[i + 1]) for _ in range(n_blocks[i + 1])]))
            setattr(self, f'adapter_norm{i}', nn.BatchNorm1d(embed_dim))
            setattr(self, f'adapter_gamma{i}', nn.Parameter(torch.ones(adapter_dim[i + 1])))
            setattr(self, f'adapter_beta{i}', nn.Parameter(torch.zeros(adapter_dim[i + 1])))

    def forward(self, x, pos, xyz, origin_x, origin_xyz, prompt_pos):
        for idx, block in enumerate(self.blocks):
            x = block(x + pos)
            if idx in self.idx:
                i = idx // self.time_step
                assert x[:, 1 + i:].size(1) == xyz.size(1)
                origin_x, origin_xyz = getattr(self, f'adapter_down{i}')(origin_x, origin_xyz)
                x_cross = getattr(self, f'adapter_norm{i}')(x[:, 1 + i:].transpose(1, 2)).transpose(1, 2)
                linear = getattr(self, f'adapter_proj{i}')
                x_cross = x_cross @ linear.weight.t()
                for adapter in getattr(self, f'adapter{i}'):
                    origin_x = adapter(x_cross, xyz, origin_x, origin_xyz)
                prompt = origin_x.mean(1, keepdim=True)
                prompt = prompt * getattr(self, f'adapter_gamma{i}') + getattr(self, f'adapter_beta{i}')
                prompt = prompt @ linear.weight
                x = torch.cat([x[:, 0:1], prompt, x[:, 1:]], dim=1)
                prompt_pos = prompt_pos.expand(prompt.size(0), -1, -1)
                pos = torch.cat([pos[:, 0:1], prompt_pos, pos[:, 1:]], dim=1)
        return x, origin_x


@MODELS.register_module()
class PointTransformerLadder(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.drop_rate = config.drop_rate
        
        ### Ladder HyperParameters ###
        self.adapter_dim = config.adapter_dim
        self.stride = config.stride
        self.r = config.r
        self.k = config.k
        self.radius = config.radius
        self.r_ratio = config.r_ratio
        self.n_blocks = config.n_blocks
        self.dropout = config.dropout
        self.prompt_length = len(self.adapter_dim) - 1

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.adapter_embedding = nn.Linear(3, self.adapter_dim[0])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.prompt_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            adapter_dim=self.adapter_dim,
            drop_rate=self.drop_rate,
            stride=self.stride,
            r=self.r,
            k=self.k,
            radius=self.radius,
            r_ratio=self.r_ratio,
            n_blocks=self.n_blocks,
            dropout=self.dropout
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.ori_norm= nn.LayerNorm(self.adapter_dim[-1])

        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2 + self.adapter_dim[-1], 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        self.build_loss_func()

        self.tfts_gamma_1, self.tfts_beta_1 = init_tfts(self.trans_dim)
        self.tfts_gamma_2, self.tfts_beta_2 = init_tfts(self.trans_dim)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.prompt_pos, std=.02)
        self.apply(self._init_weights)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('transformer_q'):
                    base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('ACT_encoder'):
                    base_ckpt[k[len('ACT_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        
        origin_x = self.adapter_embedding(pts)

        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        group_input_tokens = apply_tfts(group_input_tokens, self.tfts_gamma_1, self.tfts_beta_1)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        x, origin_x = self.blocks(x, pos, center, origin_x, pts, prompt_pos=self.prompt_pos)

        x = apply_tfts(self.norm(x), self.tfts_gamma_2, self.tfts_beta_2)
        origin_x = self.ori_norm(origin_x) 

        concat_f = torch.cat([x[:, 0] + x[:, 1: 1 + self.prompt_length].mean(1), 
                              x[:, 1 + self.prompt_length:].max(1)[0], origin_x.max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret#, concat_f
    

def init_tfts(dim):
    gamma = nn.Parameter(torch.ones(dim))
    beta = nn.Parameter(torch.zeros(dim))
    nn.init.normal_(gamma, mean=1, std=.02)
    nn.init.normal_(beta, std=.02)
    return gamma, beta


def apply_tfts(x, gamma, beta):
    assert gamma.shape == beta.shape
    if x.shape[-1] == gamma.shape[0]:
        return x * gamma + beta
    elif x.shape[1] == gamma.shape[0]:
        return x * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--leaky_relu', type=int, default=0)
    parser.add_argument('--trans_dim', type=int, default=384)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--cls_dim', type=int, default=40)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--group_size', type=int, default=32)
    parser.add_argument('--num_group', type=int, default=64)
    parser.add_argument('--encoder_dims', type=int, default=384)
    
    model = PointTransformerLadder(parser.parse_args())
    print(model)
    pts = torch.rand(2, 8192, 3)
    print(model(pts).shape)