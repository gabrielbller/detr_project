import os
import json

dataset = os.path.join("datasets", "Obstacle-detection-11")

# Caminho para o diretório de treinamento
ANNOTATION_FILE_NAME = "_annotations.coco.json"
TRAIN_DIRECTORY = os.path.join(dataset, "train")

# Carregar as anotações
with open(ANNOTATION_FILE_NAME, 'r') as f:
    annotations = json.load(f)

# Extrair as categorias
categories = annotations['categories']

# Criar o mapeamento id2label
id2label = {category['id']: category['name'] for category in categories}

print(id2label)
