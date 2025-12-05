# training/export_onnx.py
import torch
from models.efficientnet_face import EfficientNetFace


def export_onnx(
    ckpt_path: str,
    onnx_path: str = "weights/efficientnet_b0_vggface2_arcface.onnx",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(ckpt_path, map_location=device)

    backbone_name = state.get("backbone_name", "efficientnet_b0")
    embedding_dim = state.get("embedding_dim", 512)
    input_size = state.get("input_size", 224)

    model = EfficientNetFace(
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        pretrained=False,
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size, device=device)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["embedding"],
        opset_version=17,
        dynamic_axes={
            "input": {0: "batch"},
            "embedding": {0: "batch"},
        },
    )
    print(f"Exported ONNX to {onnx_path}")


if __name__ == "__main__":
    export_onnx(
        ckpt_path="weights/checkpoints/epoch_20.pth",
        onnx_path="weights/efficientnet_b0_arcface.onnx",
    )
