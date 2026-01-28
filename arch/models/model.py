
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBNAct, TransformerBlock, GatedFusion


class CNNEncoder(nn.Module):

    def __init__(self, in_ch: int = 3, base: int = 64, embed_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, base, 3, 2),
            ConvBNAct(base, base, 3, 1),
            ConvBNAct(base, base * 2, 3, 2),
            ConvBNAct(base * 2, base * 2, 3, 1),
            ConvBNAct(base * 2, embed_dim, 1, 1, p=0),
        )

    def forward(self, x):
        # x: [B,3,H,W]
        feat = self.stem(x)            # [B, C, H', W']
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # [B, N, C] where N=H'*W'
        return tokens


class Model(nn.Module):
    """
 
    """
    def __init__(
        self,
        num_classes: int,
        in_ch: int = 3,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        use_fusion: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.encoder = CNNEncoder(in_ch=in_ch, embed_dim=embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=num_heads, mlp_ratio=4.0, drop=drop, attn_drop=attn_drop)
            for _ in range(depth)
        ])

        self.use_fusion = use_fusion
        if use_fusion:
            # 
            self.fusion = GatedFusion(embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, x2=None):
        """
     
        """
        t1 = self.encoder(x)  # [B, N, C]
        for blk in self.blocks:
            t1 = blk(t1)

        if self.use_fusion:
            if x2 is None:
                raise ValueError("use_fusion=True requires x2 input.")
            t2 = self.encoder(x2)
            for blk in self.blocks:
                t2 = blk(t2)
            t = self.fusion(t1, t2)
        else:
            t = t1

        t = self.norm(t)               # [B, N, C]
        pooled = t.mean(dim=1)         # mean pooling
        logits = self.head(pooled)     # [B, num_classes]
        return logits
