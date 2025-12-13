# dataset.py
from __future__ import annotations

import os
import glob
from typing import Callable, List, Tuple, Optional, Dict, Any, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def _list_videos(folder: str) -> List[str]:
    paths = []
    if not os.path.isdir(folder):
        return paths
    # recursive
    for ext in VIDEO_EXTS:
        paths.extend(glob.glob(os.path.join(folder, "**", f"*{ext}"), recursive=True))
    paths = sorted(list(set(paths)))
    return paths


def _safe_read_frame(cap: cv2.VideoCapture, idx: int) -> Optional[np.ndarray]:
    # Seek and read one frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def _sample_indices(num_frames: int, clip_len: int, sampling: str, stride: int) -> List[int]:
    """
    sampling:
      - "random": random start, then step by stride
      - "uniform": evenly spaced indices across whole video
    """
    num_frames = max(1, int(num_frames))
    clip_len = max(1, int(clip_len))
    stride = max(1, int(stride))

    if clip_len == 1:
        if sampling == "random":
            return [np.random.randint(0, num_frames)]
        # uniform
        return [num_frames // 2]

    if sampling == "uniform":
        # evenly spaced across [0, num_frames-1]
        idxs = np.linspace(0, num_frames - 1, num=clip_len).astype(int).tolist()
        return idxs

    # random
    max_start = max(0, num_frames - 1 - (clip_len - 1) * stride)
    start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
    idxs = [min(num_frames - 1, start + i * stride) for i in range(clip_len)]
    return idxs


class VideoAntiSpoofDataset(Dataset):
    """
    Datasets for:
      root/train/real, root/train/attack
      root/val/real,   root/val/attack

    Returns:
      x: torch.FloatTensor
         - if clip_len == 1: [3,H,W]
         - else: [T,3,H,W]
      y: torch.LongTensor scalar (0=real, 1=attack)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        clip_len: int = 1,
        stride: int = 5,
        transform: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
        sampling_train: str = "random",
        sampling_val: str = "uniform",
        label_map: Optional[Dict[str, int]] = None,
        return_path: bool = False,
        return_video_id: bool = False,
    ):
        super().__init__()
        assert split in ("train", "val"), "split must be 'train' or 'val'"

        self.root = root
        self.split = split
        self.clip_len = int(clip_len)
        self.stride = int(stride)
        self.transform = transform
        self.sampling_train = sampling_train
        self.sampling_val = sampling_val
        self.return_path = return_path
        self.return_video_id = return_video_id

        self.label_map = label_map or {"real": 0, "attack": 1}

        # Collect videos per class
        items: List[Tuple[str, int]] = []
        for cls_name, cls_id in self.label_map.items():
            cls_dir = os.path.join(root, split, cls_name)
            vids = _list_videos(cls_dir)
            items.extend([(p, cls_id) for p in vids])

        if len(items) == 0:
            raise FileNotFoundError(
                f"No videos found under {os.path.join(root, split)} with classes {list(self.label_map.keys())}"
            )

        self.items = sorted(items, key=lambda x: x[0])

    def __len__(self) -> int:
        return len(self.items)

    def _default_to_tensor(self, frame_rgb: np.ndarray) -> torch.Tensor:
        # frame_rgb: H,W,3 uint8
        x = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous().float() / 255.0
        return x

    def __getitem__(self, idx: int):
        path, y = self.items[idx]

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampling = self.sampling_train if self.split == "train" else self.sampling_val
        idxs = _sample_indices(num_frames=num_frames, clip_len=self.clip_len, sampling=sampling, stride=self.stride)

        frames: List[torch.Tensor] = []
        last_ok: Optional[np.ndarray] = None

        for fi in idxs:
            frame = _safe_read_frame(cap, fi)
            if frame is None:
                frame = last_ok
            if frame is None:
                # fallback: try read first frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame0 = cap.read()
                frame = frame0 if ok else None

            if frame is None:
                cap.release()
                raise RuntimeError(f"Failed reading frames from: {path}")

            last_ok = frame

            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                x = self.transform(frame_rgb)
                if not isinstance(x, torch.Tensor):
                    raise TypeError("transform must return a torch.Tensor [3,H,W]")
            else:
                x = self._default_to_tensor(frame_rgb)

            frames.append(x)

        cap.release()

        if self.clip_len == 1:
            x_out = frames[0]  # [3,H,W]
        else:
            x_out = torch.stack(frames, dim=0)  # [T,3,H,W]

        y_out = torch.tensor(y, dtype=torch.long)

        if self.return_path or self.return_video_id:
            extra = {}
            if self.return_path:
                extra["path"] = path
            if self.return_video_id:
                extra["video_id"] = os.path.splitext(os.path.basename(path))[0]
            return x_out, y_out, extra

        return x_out, y_out


# Optional: collate nếu clip_len > 1 (stack OK vì T cố định)
def default_collate(batch):
    """
    batch item:
      (x, y) or (x, y, extra)
    """
    if len(batch[0]) == 2:
        xs, ys = zip(*batch)
        return torch.stack(xs, 0), torch.stack(ys, 0)
    xs, ys, extras = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), list(extras)
