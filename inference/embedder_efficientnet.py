import os
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort


class EfficientNetEmbedder:
    def __init__(
        self,
        model_path: str = "weights/efficientnet_b0_vggface2_arcface.onnx",
        input_size: int = 224,
        providers: Optional[list] = None,
    ):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.model_path = model_path
        self.input_size = input_size

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    # đảm bảo ảnh H×W×3 BGR, uint8
    def _ensure_bgr_uint8(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            raise ValueError("Empty image passed to EfficientNetEmbedder.")

        out = img
        if out.dtype in (np.float32, np.float64):
            out = (out * 255).clip(0, 255).astype("uint8")

        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        elif out.ndim == 3 and out.shape[2] == 1:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        if out.ndim != 3 or out.shape[2] != 3:
            raise ValueError(f"Invalid image shape {out.shape}, expected HxWx3")

        return out

    # chuẩn hóa input giống lúc training EfficientNet ([-1,1])
    def _preprocess(self, face: np.ndarray) -> np.ndarray:
        img = self._ensure_bgr_uint8(face)

        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32")
        img = img / 255.0                # [0,1]
        img = (img - 0.5) / 0.5          # [-1,1] giống train

        img = np.expand_dims(img, axis=0)  # (1,H,W,3)
        return img

    def get_embedding(self, face: np.ndarray) -> np.ndarray:
        inp = self._preprocess(face)
        outputs = self.session.run([self.output_name], {self.input_name: inp})
        emb = outputs[0][0].astype("float32")

        # normalize L2
        norm = np.linalg.norm(emb) + 1e-8
        return emb / norm


_embedder_instance: Optional[EfficientNetEmbedder] = None


def get_efficientnet_embedder() -> EfficientNetEmbedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = EfficientNetEmbedder(
            model_path="weights/efficientnet_b0_vggface2_arcface.onnx",
            input_size=224,
            providers=["CPUExecutionProvider"],
        )
    return _embedder_instance
