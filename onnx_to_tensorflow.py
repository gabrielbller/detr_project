import onnx
from onnx_tf.backend import prepare

# Carregar o modelo ONNX
onnx_model = onnx.load("detr_model.onnx")

# Converter para modelo TensorFlow
tf_rep = prepare(onnx_model)

# Salvar o modelo TensorFlow
tf_rep.export_graph("detr_tf_model")