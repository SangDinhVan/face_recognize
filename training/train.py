from collections import defaultdict
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import numpy as np
from tqdm.auto import tqdm
from torch import amp

from .datasets.dataset import VGGFace2Dataset
from .models.efficientnet_face import EfficientNetFace
from .models.arcface_head import ArcFaceHead
from .utils.checkpoint import save_checkpoint
from .utils.misc import set_seed, get_device
from .utils.init_weigth import init_weigth
from .utils.lr_schedulr import warmup_cosine_schedulr
from .utils.amp_trainer import amp_train_step
from .utils.fine_tune import freeze_backbone, unfreeze_backbone

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_config(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def evaluate_val(
    model: nn.Module,
    head: nn.Module,
    criterion: nn.Module,
    val_loader: DataLoader,
    device: str,
    use_amp: bool,
    max_pairs: int = 10000,
):
    """
    - Tính val loss (CrossEntropy trên ArcFace logits)
    - Tính verification accuracy trên val set:
        + Lấy embedding L2-norm từ model
        + Random same/diff pairs
        + Sweep threshold -> lấy acc cao nhất
    """
    model.eval()
    head.eval()

    total_loss = 0.0
    total_steps = 0

    all_embs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validating", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp and (device == "cuda")):
                # l2_norm=False cho ArcFace loss
                emb_for_loss = model(imgs, l2_norm=True)
                logits = head(emb_for_loss, labels)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            total_steps += 1

            # lấy embedding cho verification (L2-norm=True)
            emb_for_ver = model(imgs, l2_norm=True)  # [B, D]
            all_embs.append(emb_for_ver.cpu())
            all_labels.append(labels.cpu())

    val_loss = total_loss / max(1, total_steps)

    # nếu val set quá nhỏ
    if not all_embs:
        return val_loss, 0.0, 0.0

    embs = torch.cat(all_embs, dim=0).numpy()   # [N, D]
    labels = torch.cat(all_labels, dim=0).numpy().astype(np.int32)  # [N]

    N = embs.shape[0]
    if N < 2:
        return val_loss, 0.0, 0.0

    # build random verification pairs
    # ===== build BALANCED verification pairs =====

    rng = np.random.default_rng(42)
    sims = []
    gts = []

    idx_by_label = defaultdict(list)
    for idx, y in enumerate(labels):
        idx_by_label[int(y)].append(idx)

    all_labels = list(idx_by_label.keys())

    # số pair mỗi loại
    pairs_per_type = max_pairs // 2

    # ---- SAME pairs ----
    same_cnt = 0
    while same_cnt < pairs_per_type:
        y = rng.choice(all_labels)
        if len(idx_by_label[y]) < 2:
            continue
        i, j = rng.choice(idx_by_label[y], size=2, replace=False)
        sim = cosine_similarity(embs[i], embs[j])
        sims.append(sim)
        gts.append(1)
        same_cnt += 1

    # ---- DIFF pairs ----
    diff_cnt = 0
    while diff_cnt < pairs_per_type:
        y1, y2 = rng.choice(all_labels, size=2, replace=False)
        i = rng.choice(idx_by_label[y1])
        j = rng.choice(idx_by_label[y2])
        sim = cosine_similarity(embs[i], embs[j])
        sims.append(sim)
        gts.append(0)
        diff_cnt += 1


    sims = np.array(sims, dtype=np.float32)
    gts = np.array(gts, dtype=np.int32)

    # sweep threshold từ -1 -> 1
    best_acc = 0.0
    best_thr = 0.0
    for thr in np.linspace(-1.0, 1.0, 200):
        preds = (sims >= thr).astype(np.int32)
        acc = (preds == gts).mean()
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)

    return val_loss, best_acc, best_thr


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
    

    # ====== Dataset train ======
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

    # ====== Dataset val (nếu có) ======
    val_loader = None
    val_root = data_cfg.get("val_root")
    if val_root is not None and str(val_root).lower() not in ["", "none", "null"]:
        if os.path.isdir(val_root):
            val_dataset = VGGFace2Dataset(
                root_dir=val_root,
                input_size=data_cfg["input_size"],
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=train_cfg["batch_size"],
                shuffle=False,
                num_workers=train_cfg["num_workers"],
                pin_memory=True,
                drop_last=False,
            )
            print(f"Use val set from: {val_root}")
        else:
            print(f"[WARN] val_root does not exist: {val_root}, skip val.")

    # ====== Model + head ======
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

    freeze_epochs = int(train_cfg.get("freeze_epochs", 0))

    if freeze_epochs > 0:
        freeze_backbone(model)
        
    model.embedding.apply(init_weigth)
    model.bn.apply(init_weigth)

    head.apply(init_weigth)
    nn.init.normal_(head.weight, std=0.01)

    print("Backbone params:", count_params(model))
    print("Head params:", count_params(head))
    print("Total params :", count_params(model) + count_params(head))
    criterion = nn.CrossEntropyLoss()

    params = [p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad]

    optimizer = SGD(
        params,
        lr=train_cfg["base_lr"],
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )

    # scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    total_steps = train_cfg["epochs"] * len(train_loader)
    warmup_epochs = int(train_cfg.get("warmup_epochs", 2))
    warmup_steps = warmup_epochs * len(train_loader)

    scheduler = warmup_cosine_schedulr(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=float(train_cfg.get("min_lr_ratio", 0.05)),
    )


    save_dir = Path(train_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    use_amp = bool(train_cfg.get("use_amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and (device == "cuda"))

    patience = int(train_cfg.get("early_stop_patience", 0))      # 0 = tắt early stop
    min_delta = float(train_cfg.get("early_stop_min_delta", 0.0))
    
    best_ver_acc = 0.0  # chọn best theo verification accuracy
    best_train_loss = float("inf")
    no_improve_epochs = 0
    for epoch in range(1, train_cfg["epochs"] + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            unfreeze_backbone(model)

            backbone_lr = float(train_cfg.get("backbone_lr", train_cfg["base_lr"] * 0.3))
            base_lr = float(train_cfg["base_lr"])

            optimizer = SGD(
                [
                    {"params": model.backbone.parameters(), "lr": backbone_lr},
                    {"params": list(model.embedding.parameters()) + list(model.bn.parameters()), "lr": base_lr},
                    {"params": head.parameters(), "lr": base_lr},
                ],
                momentum=0.9,
                weight_decay=5e-4,
                nesterov=True,
            )

            # rebuild scheduler vì optimizer mới
            total_steps = train_cfg["epochs"] * len(train_loader)
            warmup_epochs = int(train_cfg.get("warmup_epochs", 2))
            warmup_steps = warmup_epochs * len(train_loader)
            scheduler = warmup_cosine_schedulr(optimizer, warmup_steps, total_steps, float(train_cfg.get("min_lr_ratio", 0.05)))

        model.train()
        head.train()

        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg['epochs']}")
        for step, (imgs, labels) in enumerate(pbar, start=1):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            use_autocast = use_amp and (device == "cuda")
            with amp.autocast("cuda", enabled=use_autocast):
                embeddings = model(imgs, l2_norm=False)
                logits = head(embeddings, labels)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            avg_train_loss = running_loss / step
            pbar.set_postfix(loss=f"{avg_train_loss:.4f}")

            scheduler.step()

        avg_loss = running_loss / max(1, len(train_loader))

        # ====== VALIDATION / VERIFICATION ======
        if val_loader is not None:
            val_loss, ver_acc, ver_thr = evaluate_val(
                model=model,
                head=head,
                criterion=criterion,
                val_loader=val_loader,
                device=device,
                use_amp=use_amp,
                max_pairs=5000,   # cho nhẹ, có thể tăng
            )
            print(
                f"→ [VAL] loss={val_loss:.4f} | ver_acc={ver_acc:.4f} | best_thr={ver_thr:.3f}"
            )
        else:
            val_loss, ver_acc, ver_thr = None, None, None

         
        if val_loader is not None and (ver_acc is not None):
            improved = ver_acc > (best_ver_acc + min_delta)
            if improved:
                best_ver_acc = ver_acc
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            is_best = improved
        else:
            # không có val set → fallback sang train loss
            improved = avg_loss < (best_train_loss - min_delta)
            if improved:
                best_train_loss = avg_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            is_best = improved


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
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "ver_acc": ver_acc,
            "ver_thr": ver_thr,
            "best_ver_acc": best_ver_acc,
        }

        ckpt_name = f"epoch_{epoch}.pth"
        ckpt_path = save_checkpoint(
            state=state,
            save_dir=str(save_dir),
            filename=ckpt_name,
            is_best=is_best,
            best_filename="best.pth",
        )
        print(
            f"Saved checkpoint: {ckpt_path} "
            f"(best={is_best}, train_loss={avg_loss:.4f}, best_ver_acc={best_ver_acc:.4f})"
        )

        if patience > 0 and no_improve_epochs >= patience:
            print(
                f"[Early Stop] Không cải thiện trong {patience} epoch liên tiếp. "
                f"Dừng tại epoch {epoch}."
            )
            break


def main():
    train_from_config(cfg_path="training/configs/config.yaml")


if __name__ == "__main__":
    main()
