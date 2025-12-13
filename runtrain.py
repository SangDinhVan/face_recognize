# runtrain.py
from __future__ import annotations

import argparse
import os
import yaml

from training.train2 import main as train_main


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_cfg(cfg, path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def apply_overrides(cfg, args):
    # --- paths
    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root

    if args.face_ckpt is not None:
        cfg["transfer"]["face_ckpt_path"] = args.face_ckpt

    if args.save_dir is not None:
        cfg["logging"]["save_dir"] = args.save_dir

    # --- model
    if args.backbone is not None:
        cfg["model"]["backbone_name"] = args.backbone

    if args.embedding_dim is not None:
        cfg["model"]["embedding_dim"] = int(args.embedding_dim)

    # --- train params
    if args.bs is not None:
        cfg["train"]["batch_size"] = int(args.bs)

    if args.img is not None:
        cfg["train"]["image_size"] = int(args.img)

    if args.warmup is not None:
        cfg["transfer"]["warmup_epochs"] = int(args.warmup)

    if args.finetune is not None:
        cfg["transfer"]["finetune_epochs"] = int(args.finetune)

    if args.unfreeze is not None:
        cfg["transfer"]["unfreeze_last_n_blocks"] = int(args.unfreeze)

    if args.no_amp:
        cfg["train"]["use_amp"] = False

    if args.no_class_weight:
        cfg["train"]["use_class_weight"] = False

    return cfg


def parse_args():
    p = argparse.ArgumentParser("Runner for antispoof training")

    # base
    p.add_argument("--cfg", type=str, default="training/configs/antispoofing.yaml", help="Path to yaml config")

    # common overrides
    p.add_argument("--data_root", type=str, default=None, help="Override data.root")
    p.add_argument("--face_ckpt", type=str, default=None, help="Override transfer.face_ckpt_path")
    p.add_argument("--save_dir", type=str, default=None, help="Override logging.save_dir")

    p.add_argument("--backbone", type=str, default=None, help="Override model.backbone_name (e.g. efficientnet_b0)")
    p.add_argument("--embedding_dim", type=int, default=None, help="Override model.embedding_dim")

    p.add_argument("--bs", type=int, default=None, help="Override train.batch_size")
    p.add_argument("--img", type=int, default=None, help="Override train.image_size")
    p.add_argument("--warmup", type=int, default=None, help="Override transfer.warmup_epochs")
    p.add_argument("--finetune", type=int, default=None, help="Override transfer.finetune_epochs")
    p.add_argument("--unfreeze", type=int, default=None, help="Override transfer.unfreeze_last_n_blocks")

    p.add_argument("--no_amp", action="store_true", help="Disable AMP")
    p.add_argument("--no_class_weight", action="store_true", help="Disable class weights in CE loss")

    # optional: dump merged config for reproducibility
    p.add_argument("--dump_cfg", type=str, default=None, help="Write merged config to this path then train with it")

    return p.parse_args()


def main():
    args = parse_args()

    cfg = load_cfg(args.cfg)
    cfg = apply_overrides(cfg, args)

    # If user wants, dump merged config (useful when overriding by CLI)
    if args.dump_cfg is not None:
        out = args.dump_cfg
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        save_cfg(cfg, out)
        cfg_path_to_run = out
    else:
        # write a temp merged file next to save_dir (optional but helpful)
        # to avoid touching original config, we won't write unless dump_cfg specified
        cfg_path_to_run = args.cfg

    # If dump_cfg not provided, train.py will still read from args.cfg,
    # so if you used overrides but didn't dump, we'd lose them.
    # => For safety: if any override flags are used, auto-dump to save_dir.
    used_override = any([
        args.data_root, args.face_ckpt, args.save_dir, args.backbone,
        args.embedding_dim is not None, args.bs is not None, args.img is not None,
        args.warmup is not None, args.finetune is not None, args.unfreeze is not None,
        args.no_amp, args.no_class_weight
    ])

    if used_override and args.dump_cfg is None:
        save_dir = cfg["logging"]["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        auto_cfg = os.path.join(save_dir, "merged_config.yaml")
        save_cfg(cfg, auto_cfg)
        cfg_path_to_run = auto_cfg
        print(f"[runtrain] Overrides detected -> wrote merged config to: {auto_cfg}")

    train_main(cfg_path_to_run)


if __name__ == "__main__":
    main()
