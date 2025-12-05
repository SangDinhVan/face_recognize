import os
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort


class FaceEmbedder:
    def __init__(
        self,
        model_path: str = "weights/arcface_r100.onnx",
        input_size: int = 112,
        providers: Optional[list] = None,
    ):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at: {model_path}")

        self.model_path = model_path
        self.input_size = input_size

        self.session = ort.InferenceSession(self.model_path, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _ensure_bgr_uint8(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            raise ValueError("Empty image passed to FaceEmbedder.")

        out = img
        if out.dtype == np.float32 or out.dtype == np.float64:
            out = (out * 255).clip(0, 255).astype("uint8")

        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        elif out.ndim == 3 and out.shape[2] == 1:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        if out.ndim != 3 or out.shape[2] != 3:
            raise ValueError(f"Invalid image shape: {out.shape}, expected HxWx3")

        return out

    def _preprocess(self, face: np.ndarray) -> np.ndarray:
        img = self._ensure_bgr_uint8(face)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32")
        img = (img - 127.5) / 128.0
        return np.expand_dims(img, axis=0)

    def get_embedding(self, face: np.ndarray) -> np.ndarray:
        inp = self._preprocess(face)
        outputs = self.session.run([self.output_name], {self.input_name: inp})
        emb = outputs[0][0].astype("float32")
        return emb / (np.linalg.norm(emb) + 1e-8)


_embedder_instance = None


def get_face_embedder() -> FaceEmbedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = FaceEmbedder(
            model_path="weights/arcface_r100.onnx",
            input_size=112,
            providers=["CPUExecutionProvider"],
        )
    return _embedder_instance
