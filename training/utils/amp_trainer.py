import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

def amp_train_step(
    model,
    arcface,
    images,
    labels,
    optimizer,
    scaler: GradScaler,
    clip_grad=1.0
):
    optimizer.zero_grad(set_to_none=True)

    with autocast():
        emb = model(images, l2_norm=False)
        logits = arcface(emb, labels)
        loss = F.cross_entropy(logits, labels)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    if clip_grad > 0:
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(arcface.parameters()),
            clip_grad
        )

    scaler.step(optimizer)
    scaler.update()

    return loss.item()
