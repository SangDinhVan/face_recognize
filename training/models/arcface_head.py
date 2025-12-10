import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.normal_(self.weight, std=0.01)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb, labels):
        # Only normalize here
        emb = F.normalize(emb, dim=1)
        W = F.normalize(self.weight, dim=1)

        cosine = F.linear(emb, W)

        # stable cosine
        sine = torch.sqrt(1.0 - cosine * cosine + 1e-7)

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine, device=cosine.device)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits *= self.s
        return logits

