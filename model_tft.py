import torch
import einops as ep
import torch.nn as nn
import torch.nn.functional as ff
from torch import einsum


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    # dim: 输入信号的嵌入维度
    # dim_head: qkv的维度，即嵌入后的信号经qkv权重计算后的维度
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5  # score归一化，即根号dk

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _ = x.shape
        # print('b:', b, 'n:', n, '_:', _)
        h = self.heads
        # print('h:', h)
        # qkv = self.to_qkv(x).chunk(3, dim=-1)
        qkv = self.to_qkv(x)
        qkv = qkv.chunk(3, dim=-1)
        # print('qkv:', len(qkv))
        q, k, v = map(lambda t: ep.rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # print('q:', q.shape, 'k:', k.shape, 'v:', v.shape)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale    # 仅仅是点积操作（先转置后点积，使用einsum避免了操作）
        # print('scale:', self.scale)
        # print('dots:', dots.shape)

        attn = self.attend(dots)
        # print('attn:', attn.shape)  # attn: torch.Size([10, 15, 21, 21])
        # print('v:', v.shape)  # v: torch.Size([10, 15, 21, 64])

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # print('out:', out.shape)  # out: torch.Size([10, 15, 21, 64])
        out = ep.rearrange(out, 'b h n d -> b n (h d)')  # 把多头直接拼接在一起，然后用一个附加权重于贞与它们相乘，生成原始大小Z矩阵
        # print('out:', out.shape)  # out: torch.Size([10, 21, 960])
        # temp = self.to_out(out)
        # print('temp:', temp.shape)  # temp: torch.Size([10, 21, 1024])
        # # 为什么从960维度变到1024维度了？？
        # # 前馈神经网络接受的是1个矩阵（其中每行的向量表示一个词），即前馈神经网络的输出与输出形状均与词向量形状保持一致
        # return temp
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, get_features=False):
        features = []
        for attn, ff in self.layers:
            x = attn(x) + x
            # attn: 多头自注意力  x: 原始输入
            # print('attn:', x.shape)
            x = ff(x) + x
            # print('ff:', x.shape)
            # 在经过一次Transformer循环之后，变量shape和加入分类头后的输入数据shape一样
            if get_features:
                features.append(x)
        if get_features:
            return x, features
        else:
            return x


class TFT(nn.Module):
    def __init__(self, *, num_classes, depth, heads, dim=768, dim_head=64, mlp_dim=4096, pool='cls',
                 simple_length=640, data_length=16, dropout=0., emb_dropout=0.):
        super().__init__()
        # dim:词嵌入的维度
        # heads:注意力的头数
        # dim_heads:注意力机制qkv的维度
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        data_dimensionality = int(simple_length / data_length)

        self.to_patch_embedding = nn.Sequential(nn.Linear(data_length, dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, data_dimensionality + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # 归一化
            nn.Linear(dim, num_classes)
        )

    def forward(self, signal, transfer=False, get_features=False):
        features = []

        if transfer:
            signal_length = signal.shape[2]
            signal = ep.rearrange(signal, 'b p l -> b ( ) (p l)')
            signal = ff.interpolate(
                signal, scale_factor=640 / 512, mode='linear', align_corners=True, recompute_scale_factor=True
            )
            signal = ep.rearrange(signal, 'b i (p l) -> b (p i) l', l=signal_length)

        x = self.to_patch_embedding(signal)
        b, n, _ = x.shape

        cls_tokens = ep.repeat(self.cls_token, '() n d -> b n d', b=b)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        if get_features:
            x, features = self.transformer(x, get_features)
        else:
            x = self.transformer(x, get_features)
        # print(x.shape)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # 提取分类头  torch.Size([5, 1024])
        # print(x.shape)

        x = self.to_latent(x)
        # print(x.shape)

        x = self.mlp_head(x)

        if get_features:
            return x, features
        else:
            return x

    def forward_with_features(self, signal, transfer=True):
        return self.forward(signal, transfer, get_features=True)
