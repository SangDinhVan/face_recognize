import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm


def detect_biggest_face(
    model: YOLO,
    img: Image.Image,
    conf_thres: float = 0.3
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect face bằng YOLOv8-face, trả về bbox lớn nhất (x1, y1, x2, y2).
    Nếu không thấy face thì trả về None.
    """
    # PIL -> numpy (RGB)
    img_np = np.array(img)

    results = model.predict(img_np, conf=conf_thres, verbose=False)
    if len(results) == 0:
        return None

    boxes = results[0].boxes
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()  # shape: (N, 4)

    # chọn box có diện tích lớn nhất
    max_area = 0
    best_box = None
    for x1, y1, x2, y2 in xyxy:
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area = w * h
        if area > max_area:
            max_area = area
            best_box = (int(x1), int(y1), int(x2), int(y2))

    return best_box


def expand_box(
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    scale: float = 1.2
) -> Tuple[int, int, int, int]:
    """
    Nới rộng bbox một chút để lấy cả vùng quanh mặt.
    scale > 1 nghĩa là mở rộng.
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale

    new_x1 = int(max(0, cx - w / 2.0))
    new_y1 = int(max(0, cy - h / 2.0))
    new_x2 = int(min(img_w - 1, cx + w / 2.0))
    new_y2 = int(min(img_h - 1, cy + h / 2.0))

    return new_x1, new_y1, new_x2, new_y2


def preprocess_dataset(
    src_root: str,
    dst_root: str,
    model_path: str,
    output_size: int = 224,
    conf_thres: float = 0.3,
    overwrite: bool = False
):
    """
    src_root: thư mục gốc raw data (cấu trúc: root/id/*.jpg)
    dst_root: thư mục output (giữ nguyên cấu trúc id)
    model_path: đường dẫn file .pt YOLOv8-face (ví dụ: yolov8n-face.pt)
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    print(f"[Preprocess] Load YOLO model from: {model_path}")
    model = YOLO(model_path)

    # liệt kê các class (id)
    id_dirs = sorted([d for d in src_root.iterdir() if d.is_dir()])
    print(f"[Preprocess] Identities: {len(id_dirs)}")

    for id_dir in id_dirs:
        rel_id = id_dir.name
        out_id_dir = dst_root / rel_id
        out_id_dir.mkdir(parents=True, exist_ok=True)

        img_paths = [
            p for p in id_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
        if len(img_paths) == 0:
            continue

        print(f"[Preprocess] ID={rel_id}, images={len(img_paths)}")

        for img_path in tqdm(img_paths, desc=f"{rel_id}", leave=False):
            out_path = out_id_dir / img_path.name

            if out_path.exists() and not overwrite:
                # đã xử lý rồi thì bỏ qua
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[Warning] Cannot open {img_path}: {e}")
                continue

            w, h = img.size

            box = detect_biggest_face(model, img, conf_thres=conf_thres)
            if box is None:
                # không detect được mặt -> có thể bỏ hoặc copy nguyên ảnh
                # ở đây mình bỏ qua cho sạch
                # Nếu muốn lưu img resize luôn thì thay đổi ở đây
                # face = img.resize((output_size, output_size))
                # face.save(out_path)
                continue

            # nới rộng box một chút
            box = expand_box(box, w, h, scale=1.2)

            x1, y1, x2, y2 = box
            face = img.crop((x1, y1, x2, y2))

            # resize về kích thước input của model
            face = face.resize((output_size, output_size), Image.BILINEAR)

            # "xử lý sương sương": có thể thêm blur nhẹ/auto contrast nếu muốn
            # ví dụ:
            # from PIL import ImageFilter, ImageOps
            # face = ImageOps.autocontrast(face)

            try:
                face.save(out_path)
            except Exception as e:
                print(f"[Warning] Cannot save {out_path}: {e}")
                continue

    print("[Preprocess] Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess face dataset with YOLOv8-face"
    )
    parser.add_argument("--src_root", type=str, required=True,
                        help="Thư mục dữ liệu raw, dạng root/id/*.jpg")
    parser.add_argument("--dst_root", type=str, required=True,
                        help="Thư mục lưu ảnh mặt đã crop")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Đường dẫn file YOLOv8-face .pt")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Kích thước output (H=W)")
    parser.add_argument("--conf_thres", type=float, default=0.3,
                        help="Confidence threshold cho YOLO")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ghi đè nếu file output đã tồn tại")
    return parser.parse_args()


if __name__ == "__main__":
    preprocess_dataset(
        src_root="data/vggface2/val",
        dst_root="data/vggface2_processed/val",
        model_path="weights/yolov8n-face-lindevs.onnx",
        output_size=112,
        conf_thres=0.3,
        overwrite=False
    )

