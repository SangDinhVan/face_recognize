import os
from pathlib import Path
from typing import List, Tuple, Dict

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class VGGFace2Dataset(Dataset):

    def __init__(self, root_dir: str, input_size: int = 224):
        self.root_dir = Path(root_dir)
        self.input_size = input_size

        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx: Dict[str, int] = {}

    
        classes = sorted(
            [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        )
        for idx, cls_name in enumerate(classes):
            self.class_to_idx[cls_name] = idx
            cls_dir = self.root_dir / cls_name
            for fname in os.listdir(cls_dir):
                fpath = cls_dir / fname
                if fpath.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((fpath, idx))

        self.transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),  
        ])

        print(f"[VGGFace2Dataset] root={self.root_dir}")
        print(f"  identities: {len(self.class_to_idx)}")
        print(f"  images    : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label
