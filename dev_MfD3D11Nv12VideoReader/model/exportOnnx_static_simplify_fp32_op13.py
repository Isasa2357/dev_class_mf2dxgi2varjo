from ultralytics import YOLO
import shutil

model = YOLO("best_n_300.pt")

model.export(
    format="onnx",
    imgsz=640,
    batch=1,
    dynamic=False,
    simplify=True,
    half=False,
    opset=13,
    nms=False,
    device="cpu",
)

shutil.copy("best_n_300.onnx", "best_n_300_static_simplify_fp32_op13.onnx")