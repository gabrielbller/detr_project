import torch
from transformers import DetrForObjectDetection
import numpy as np

# Carregar o modelo salvo
MODEL_PATH = "outputs_bin"  # Substitua pelo caminho do seu modelo salvo
model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
model.eval()  # Colocar o modelo em modo de avaliação

# Configuração do dummy input
batch_size = 1
num_channels = 3
height, width = 1334, 1334  # Use as dimensões máximas ou ajuste conforme necessário

# Criar entradas fictícias (dummy inputs)
dummy_pixel_values = torch.randn(batch_size, num_channels, height, width, dtype=torch.float32)
dummy_pixel_mask = torch.ones(batch_size, height, width, dtype=torch.float32)

# Especificar nomes de entradas e saídas
input_names = ["pixel_values", "pixel_mask"]
output_names = ["logits", "pred_boxes"]

# Exportar para ONNX
onnx_model_path = "detr_model.onnx"
torch.onnx.export(
    model,                                 # Modelo PyTorch
    (dummy_pixel_values, dummy_pixel_mask),# Entradas do modelo
    onnx_model_path,                       # Caminho para salvar o modelo ONNX
    export_params=True,                    # Exportar parâmetros treinados
    opset_version=11,                      # Versão do opset do ONNX
    input_names=input_names,               # Nomes das entradas
    output_names=output_names,             # Nomes das saídas
    dynamic_axes={
        "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
        "pixel_mask": {0: "batch_size", 1: "height", 2: "width"},
        "logits": {0: "batch_size"},
        "pred_boxes": {0: "batch_size"}
    }
)

print(f"Modelo exportado para {onnx_model_path}")