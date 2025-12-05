import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    state: Dict[str, Any],
    save_dir: str,
    filename: str = "last.pth",
    is_best: bool = False,
    best_filename: str = "best.pth",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = save_dir / filename
    torch.save(state, ckpt_path)

    if is_best:
        best_path = save_dir / best_filename
        torch.save(state, best_path)

    return str(ckpt_path)


def load_checkpoint(
    ckpt_path: str,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location=map_location or "cpu")


def get_latest_checkpoint(save_dir: str, prefix: str = "epoch_") -> Optional[str]:

    save_dir = Path(save_dir)
    if not save_dir.exists():
        return None

    ckpts = sorted(save_dir.glob(f"{prefix}*.pth"))
    if not ckpts:
        return None
    return str(ckpts[-1])
