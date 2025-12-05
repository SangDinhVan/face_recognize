import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        s: float = 64.0,
        m: float = 0.5,
    ):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # normalize
        embeddings = F.normalize(embeddings)
        W = F.normalize(self.weight)

        cosine = F.linear(embeddings, W)  # [B, C]
        sine = torch.sqrt((1.0 - torch.clamp(cosine**2, 0, 1)))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits *= self.s
        return logits
