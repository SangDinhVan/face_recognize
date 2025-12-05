# training/train_vggface2.py
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

from datasets.dataset import VGGFace2Dataset
from models.efficientnet_face import EfficientNetFace
from models.arcface_head import ArcFaceHead
from utils.checkpoint import save_checkpoint
from utils.misc import set_seed, get_device


def load_config(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def train_from_config(cfg_path: str = "training/configs/config.yaml"):
    cfg = load_config(cfg_path)

    exp_name = cfg.get("experiment_name", "exp")
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    arcface_cfg = cfg.get("arcface", {})

    set_seed(42)
    device = get_device()
    print(f"Experiment: {exp_name}")
    print(f"Device    : {device}")

    # Dataset
    train_dataset = VGGFace2Dataset(
        root_dir=data_cfg["train_root"],
        input_size=data_cfg["input_size"],
    )
    num_classes = len(train_dataset.class_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # Model + head
    model = EfficientNetFace(
        backbone_name=model_cfg["backbone_name"],
        embedding_dim=model_cfg["embedding_dim"],
        pretrained=True,
    ).to(device)

    head = ArcFaceHead(
        embedding_dim=model_cfg["embedding_dim"],
        num_classes=num_classes,
        s=arcface_cfg.get("scale_s", 64.0),
        m=arcface_cfg.get("margin_m", 0.5),
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = SGD(
        list(model.parameters()) + list(head.parameters()),
        lr=train_cfg["base_lr"],
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    save_dir = Path(train_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    use_amp = bool(train_cfg.get("use_amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and (device == "cuda"))

    best_loss = float("inf")

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        head.train()

        running_loss = 0.0

        for step, (imgs, labels) in enumerate(train_loader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp and (device == "cuda")):
                embeddings = model(imgs, l2_norm=False)
                logits = head(embeddings, labels)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())

            if step % 50 == 0:
                print(
                    f"[Epoch {epoch}/{train_cfg['epochs']}] "
                    f"Step {step}/{len(train_loader)} "
                    f"Loss: {running_loss / step:.4f}"
                )

        scheduler.step()

        avg_loss = running_loss / max(1, len(train_loader))
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        state = {
            "epoch": epoch,
            "experiment_name": exp_name,
            "model": model.state_dict(),
            "head": head.state_dict(),
            "num_classes": num_classes,
            "backbone_name": model_cfg["backbone_name"],
            "embedding_dim": model_cfg["embedding_dim"],
            "input_size": data_cfg["input_size"],
            "train_cfg": train_cfg,
            "arcface_cfg": arcface_cfg,
        }

        ckpt_name = f"epoch_{epoch}.pth"
        ckpt_path = save_checkpoint(
            state=state,
            save_dir=str(save_dir),
            filename=ckpt_name,
            is_best=is_best,
            best_filename="best.pth",
        )
        print(f"Saved checkpoint: {ckpt_path}  (best={is_best}, avg_loss={avg_loss:.4f})")


def main():
    train_from_config(cfg_path="training/configs/config.yaml")

if __name__ == "__main__":
    main()