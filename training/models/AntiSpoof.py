# model.py
from __future__ import annotations

from collections import OrderedDict
from typing import Tuple, List, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class EfficientNetAntiSpoof(nn.Module):
    """
    Transfer từ EfficientNetFace (backbone + embedding + bn) và thêm classifier cho anti-spoof.
    - Nếu bạn load ckpt từ EfficientNetFace: dùng load_face_ckpt(..., strict=False).
    - Classifier là layer mới (random init) => phải train.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        embedding_dim: int = 512,
        pretrained_backbone: bool = False,  # set False nếu bạn sẽ load ckpt của bạn
        num_classes: int = 2,               # 2-class: real/fake
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained_backbone,
            num_classes=0,
            global_pool="avg",
        )

        feat_dim = self.backbone.num_features
        self.embedding = nn.Linear(feat_dim, embedding_dim, bias=False)
        self.bn = nn.BatchNorm1d(embedding_dim)

        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, l2_norm: bool = True, return_emb: bool = False):
        feat = self.backbone(x)            # [B, feat_dim]
        emb = self.embedding(feat)         # [B, D]
        emb = self.bn(emb)

        if l2_norm:
            emb = F.normalize(emb, p=2, dim=1)

        logits = self.classifier(self.drop(emb))  # [B, num_classes]
        if return_emb:
            return logits, emb
        return logits


# -------------------------
# Load checkpoint (từ model face cũ)
# -------------------------
def _strip_prefix(state: Dict[str, torch.Tensor], prefix: str = "module.") -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state.keys()):
        return state
    new_state = OrderedDict()
    for k, v in state.items():
        new_state[k[len(prefix):]] = v
    return new_state


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """
    Hỗ trợ các format ckpt phổ biến:
    - torch.save(model.state_dict())
    - {"state_dict": ...}
    - {"model": ...}
    - {"net": ...}
    """
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "net"):
            if key in ckpt and isinstance(ckpt[key], (dict, OrderedDict)):
                return ckpt[key]
    # fallback: assume ckpt itself is state_dict
    if isinstance(ckpt, (dict, OrderedDict)):
        return ckpt
    raise ValueError("Unsupported checkpoint format. Expected a state_dict or dict containing it.")


def load_face_ckpt(
    model: nn.Module,
    ckpt_path: str,
    device: Union[str, torch.device] = "cpu",
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Load weight từ EfficientNetFace sang EfficientNetAntiSpoof.
    strict=False là chuẩn (vì classifier mới sẽ missing keys).
    Return: (missing_keys, unexpected_keys)
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state = _extract_state_dict(ckpt)
    state = _strip_prefix(state, "module.")
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return missing, unexpected


# -------------------------
# Freeze / unfreeze utils
# -------------------------
def freeze_all_except_classifier(model: EfficientNetAntiSpoof) -> None:
    """Warmup: chỉ train classifier."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_last_n_blocks_efficientnet(model: EfficientNetAntiSpoof, n_blocks: int = 2, unfreeze_embed: bool = True) -> None:
    """
    Fine-tune: mở n block cuối của EfficientNet backbone.
    Lưu ý: timm EfficientNet thường có model.backbone.blocks
    """
    # luôn mở classifier
    for p in model.classifier.parameters():
        p.requires_grad = True

    if hasattr(model.backbone, "blocks"):
        total = len(model.backbone.blocks)
        start = max(0, total - n_blocks)
        for i in range(start, total):
            for p in model.backbone.blocks[i].parameters():
                p.requires_grad = True
    else:
        # fallback: nếu backbone không có blocks thì mở toàn bộ backbone
        for p in model.backbone.parameters():
            p.requires_grad = True

    if unfreeze_embed:
        for p in model.embedding.parameters():
            p.requires_grad = True
        for p in model.bn.parameters():
            p.requires_grad = True


def freeze_batchnorm(model: nn.Module) -> None:
    """
    Data ít => BN dễ lệch. Đặt BN eval + khóa params.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
