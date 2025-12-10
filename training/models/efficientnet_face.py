import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class EfficientNetFace(nn.Module):
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        embedding_dim: int = 512,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,   
            global_pool="avg",
        )

        feat_dim = self.backbone.num_features

        self.embedding = nn.Linear(feat_dim, embedding_dim, bias=False)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x, l2_norm: bool = True):
        feat = self.backbone(x)        # [B, feat_dim]
        emb = self.embedding(feat)     # [B, D]
        emb = self.bn(emb)
        if l2_norm:
            emb = F.normalize(emb, p=2, dim=1)
        return emb