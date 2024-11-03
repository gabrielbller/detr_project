import torch
from transformers import DetrForObjectDetection

# Diretório onde o modelo está salvo
model_directory = 'outputs'

# Carregar o modelo a partir do diretório
model = DetrForObjectDetection.from_pretrained(model_directory)

# Salvar o modelo no formato .pt (apenas state_dict)
torch.save(model.state_dict(), 'detr_model.pt')

# Opcional: Salvar o modelo completo
torch.save(model, 'detr_model_complete.pt')
