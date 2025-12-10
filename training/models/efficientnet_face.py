import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class EfficientNetFace(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", embedding_dim=512, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        feat_dim = self.backbone.num_features

        self.embedding = nn.Linear(feat_dim, embedding_dim, bias=False)
        self.bn = nn.BatchNorm1d(embedding_dim, affine=True)

    def forward(self, x):
        feat = self.backbone(x)
        emb = self.embedding(feat)
        emb = self.bn(emb)    
        return emb

