# train.py
from __future__ import annotations

import os
import math
import time
import yaml
import random
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms

from .datasets.videodataset import VideoAntiSpoofDataset
from .models.AntiSpoof import (
    EfficientNetAntiSpoof,
    load_face_ckpt,
    freeze_all_except_classifier,
    unfreeze_last_n_blocks_efficientnet,
    freeze_batchnorm,
)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_ckpt(path: str, model: nn.Module, epoch: int, best_score: float, cfg: Dict[str, Any]):
    ckpt = {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "cfg": cfg,
    }
    torch.save(ckpt, path)


def build_transforms(image_size: int, is_train: bool):
    # Dataset trả numpy RGB HWC; ToPILImage nhận numpy ok
    if is_train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.6),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])


def build_ce_loss(cfg: Dict[str, Any], device: torch.device):
    if not cfg["train"].get("use_class_weight", True):
        return nn.CrossEntropyLoss()

    tr_real = float(cfg["counts"]["train_real"])
    tr_attk = float(cfg["counts"]["train_attack"])
    # label: real=0, attack=1
    counts = torch.tensor([tr_real, tr_attk], dtype=torch.float32, device=device)
    w = counts.sum() / (2.0 * counts)
    return nn.CrossEntropyLoss(weight=w)


def build_optimizer(model: nn.Module, lr_head: float, lr_backbone: float, weight_decay: float):
    head_params = list(model.classifier.parameters())
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("classifier.")]
    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": lr_backbone})
    groups.append({"params": head_params, "lr": lr_head})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


@torch.no_grad()
def eval_frame_and_video(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    loader phải return (x, y, extra) với extra có video_id.
    Tính:
      - frame_acc
      - video_acc (aggregate mean prob per video)
    """
    model.eval()
    total, correct = 0, 0

    # video aggregation
    # store list probs per video_id
    vid_probs: Dict[str, List[float]] = {}
    vid_label: Dict[str, int] = {}

    for batch in loader:
        if len(batch) == 2:
            x, y = batch
            extra = None
        else:
            x, y, extra = batch

        x = x.to(device)
        y = y.to(device)

        logits = model(x)  # [B,2]
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(attack)
        pred = torch.argmax(logits, dim=1)

        total += y.numel()
        correct += (pred == y).sum().item()

        if extra is not None:
            for i, ex in enumerate(extra):
                if isinstance(ex, dict):
                    vid = ex.get("video_id") or ex.get("path") or f"idx_{i}"
                else:
                    # ex là string (path)
                    vid = ex

                vid_probs.setdefault(vid, []).append(float(probs[i].detach().cpu().item()))
                vid_label[vid] = int(y[i].detach().cpu().item())

    frame_acc = correct / max(1, total)

    # video-level: mean prob > 0.5 => attack
    v_total, v_correct = 0, 0
    if len(vid_probs) > 0:
        for vid, ps in vid_probs.items():
            p_mean = float(np.mean(ps))
            v_pred = 1 if p_mean > 0.5 else 0
            v_y = vid_label[vid]
            v_total += 1
            v_correct += int(v_pred == v_y)
        video_acc = v_correct / max(1, v_total)
    else:
        video_acc = float("nan")

    return frame_acc, video_acc


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device: torch.device,
                    use_amp: bool, grad_clip: Optional[float], print_every: int):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    total_loss, total, correct = 0.0, 0, 0
    t0 = time.time()

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * y.size(0)
        pred = torch.argmax(logits.detach(), dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

        if print_every and step % print_every == 0:
            dt = time.time() - t0
            avg_loss = total_loss / max(1, total)
            avg_acc = correct / max(1, total)
            print(f"  step {step:5d} | loss {avg_loss:.4f} | acc {avg_acc:.4f} | {dt:.1f}s")

    return total_loss / max(1, total), correct / max(1, total)


def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = cfg["logging"]["save_dir"]
    ensure_dir(save_dir)

    # ---- Datasets
    img_size = int(cfg["train"]["image_size"])
    train_tf = build_transforms(img_size, is_train=True)
    val_tf = build_transforms(img_size, is_train=False)

    label_map = cfg.get("labels", {"real": 0, "attack": 1})

    train_ds = VideoAntiSpoofDataset(
        root=cfg["data"]["root"],
        split=cfg["data"]["train"]["split"],
        clip_len=int(cfg["data"]["train"]["clip_len"]),
        stride=int(cfg["data"]["train"]["stride"]),
        transform=train_tf,
        label_map=label_map,
        return_path=False,
        return_video_id=False,
    )

    # val: cần video_id để aggregate
    val_ds = VideoAntiSpoofDataset(
        root=cfg["data"]["root"],
        split=cfg["data"]["val"]["split"],
        clip_len=int(cfg["data"]["val"]["clip_len"]),
        stride=int(cfg["data"]["val"]["stride"]),
        transform=val_tf,
        label_map=label_map,
        return_path=True,
        return_video_id=True,
    )

    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["data"]["num_workers"])

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True, drop_last=False
    )

    # ---- Model
    mcfg = cfg["model"]
    model = EfficientNetAntiSpoof(
        backbone_name=mcfg["backbone_name"],
        embedding_dim=int(mcfg["embedding_dim"]),
        pretrained_backbone=bool(mcfg.get("pretrained_backbone", False)),
        num_classes=int(mcfg.get("num_classes", 2)),
        dropout=float(mcfg.get("dropout", 0.2)),
    ).to(device)

    # ---- Load face ckpt (transfer từ cái bạn có sẵn)
    ckpt_path = cfg["transfer"]["face_ckpt_path"]
    if ckpt_path and os.path.isfile(ckpt_path):
        missing, unexpected = load_face_ckpt(model, ckpt_path, device=device, strict=False)
        print("[Transfer] loaded face ckpt.")
        print("  missing:", missing)
        print("  unexpected:", unexpected)
        print("  (missing classifier.* là bình thường)")
    else:
        print(f"[Transfer] WARNING: face_ckpt_path not found: {ckpt_path} (train from scratch classifier/backbone init)")

    # ---- Loss
    criterion = build_ce_loss(cfg, device=device)

    # ---- Training setup
    use_amp = bool(cfg["train"].get("use_amp", True))
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    wd = float(cfg["train"].get("weight_decay", 1e-4))
    print_every = int(cfg["logging"].get("print_every", 50))

    best_video_acc = -1.0
    best_path = os.path.join(save_dir, "best.pth")

    # =========================
    # Stage 1: Warmup (train classifier only)
    # =========================
    print("\n=== Stage 1: Warmup (classifier only) ===")
    freeze_all_except_classifier(model)

    if cfg["transfer"].get("freeze_bn", True):
        freeze_batchnorm(model)

    opt = build_optimizer(
        model,
        lr_head=float(cfg["train"]["lr_head_warmup"]),
        lr_backbone=float(cfg["train"]["lr_backbone_warmup"]),
        weight_decay=wd,
    )

    warmup_epochs = int(cfg["transfer"]["warmup_epochs"])
    for epoch in range(1, warmup_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, opt, criterion, device=device,
            use_amp=use_amp, grad_clip=grad_clip, print_every=print_every
        )
        frame_acc, video_acc = eval_frame_and_video(model, val_loader, device=device)
        print(f"[Warmup {epoch}/{warmup_epochs}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | val frame_acc={frame_acc:.4f} video_acc={video_acc:.4f}")

        if video_acc > best_video_acc:
            best_video_acc = video_acc
            if cfg["logging"].get("save_best", True):
                save_ckpt(best_path, model, epoch=epoch, best_score=best_video_acc, cfg=cfg)

    # =========================
    # Stage 2: Fine-tune (unfreeze last blocks)
    # =========================
    print("\n=== Stage 2: Fine-tune (last blocks) ===")
    unfreeze_last_n_blocks_efficientnet(
        model,
        n_blocks=int(cfg["transfer"]["unfreeze_last_n_blocks"]),
        unfreeze_embed=bool(cfg["transfer"].get("unfreeze_embed", True)),
    )

    if cfg["transfer"].get("freeze_bn", True):
        freeze_batchnorm(model)

    opt = build_optimizer(
        model,
        lr_head=float(cfg["train"]["lr_head_finetune"]),
        lr_backbone=float(cfg["train"]["lr_backbone_finetune"]),
        weight_decay=wd,
    )

    finetune_epochs = int(cfg["transfer"]["finetune_epochs"])
    for epoch in range(1, finetune_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, opt, criterion, device=device,
            use_amp=use_amp, grad_clip=grad_clip, print_every=print_every
        )
        frame_acc, video_acc = eval_frame_and_video(model, val_loader, device=device)
        print(f"[Finetune {epoch}/{finetune_epochs}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | val frame_acc={frame_acc:.4f} video_acc={video_acc:.4f}")

        if video_acc > best_video_acc:
            best_video_acc = video_acc
            if cfg["logging"].get("save_best", True):
                save_ckpt(best_path, model, epoch=warmup_epochs + epoch, best_score=best_video_acc, cfg=cfg)

    print("\nDone.")
    print("Best val video_acc:", best_video_acc)
    if cfg["logging"].get("save_best", True):
        print("Saved best to:", best_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/antispoofing.yaml")
    args = parser.parse_args()
    main(args.cfg)
