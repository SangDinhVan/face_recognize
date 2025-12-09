# training/export_onnx.py
import torch
import torch.nn as nn

from models.efficientnet_face import EfficientNetFace  # giữ nguyên như bạn đang dùng


class NHWCWrapper(nn.Module):
    """
    Wrapper để ONNX nhận input NHWC (N, H, W, C).
    Bên trong sẽ permute sang NCHW (N, C, H, W) trước khi đưa vào EfficientNetFace.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        # x: (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        # EfficientNetFace.forward mặc định l2_norm=True -> emb L2-normalized
        emb = self.backbone(x)
        return emb


def export_onnx(
    ckpt_path: str = "weights/checkpoints/best.pth",
    onnx_path: str = "weights/efficientnet_arcface.onnx",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load checkpoint
    state = torch.load(ckpt_path, map_location=device)
    backbone_name = state.get("backbone_name", "efficientnet_b0")
    embedding_dim = state.get("embedding_dim", 512)
    input_size = state.get("input_size", 112)

    # tạo backbone
    backbone = EfficientNetFace(
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        pretrained=False,   # load state_dict nên không cần pretrained
    ).to(device)
    backbone.load_state_dict(state["model"], strict=True)
    backbone.eval()

    # bọc lại để nhận NHWC
    model = NHWCWrapper(backbone).to(device)
    model.eval()

    # dummy input: (N, H, W, C) = NHWC
    dummy = torch.randn(1, input_size, input_size, 3, device=device)

    # export onnx
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],        # (N, H, W, C)
        output_names=["embedding"],   # (N, D)
        opset_version=17,
        dynamic_axes={
            "input": {0: "batch"},
            "embedding": {0: "batch"},
        },
        do_constant_folding=True,
    )

    print(f"Exported ONNX to {onnx_path}")
    print(f"- backbone : {backbone_name}")
    print(f"- emb_dim  : {embedding_dim}")
    print(f"- input_sz : (N, {input_size}, {input_size}, 3)  [NHWC]")


if __name__ == "__main__":
    export_onnx(
        ckpt_path="weights/checkpoints/best.pth",
        onnx_path="weights/efficientnet_arcface.onnx",
    )
