import onnx

onnx_model = onnx.load("detr_model.onnx")
onnx.checker.check_model(onnx_model)