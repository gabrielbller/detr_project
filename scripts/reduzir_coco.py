import shutil
import os
import json

# Diretório original das imagens
SOURCE_IMAGE_DIR = 'data/train2017'

# Diretório para salvar as imagens filtradas
DESTINATION_IMAGE_DIR = 'data/train2017_filtered'

# Carregar as imagens filtradas da Parte 1
with open('data/annotations/instances_train2017_filtered.json', 'r') as f:
    filtered_coco_data = json.load(f)

filtered_images = filtered_coco_data['images']

# Criar o diretório de destino se não existir
os.makedirs(DESTINATION_IMAGE_DIR, exist_ok=True)

# Copiar as imagens
for img in filtered_images:
    file_name = img['file_name']
    src_path = os.path.join(SOURCE_IMAGE_DIR, file_name)
    dst_path = os.path.join(DESTINATION_IMAGE_DIR, file_name)
    if os.path.exists(src_path):
        shutil.copyfile(src_path, dst_path)
    else:
        print(f"Imagem não encontrada: {src_path}")

print(f"Imagens copiadas para {DESTINATION_IMAGE_DIR}")
