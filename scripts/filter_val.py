import os
import json
import shutil
from tqdm import tqdm

# --- Configurações ---

# Caminhos para os diretórios e arquivos de validação
original_annotations_path = 'data/annotations/instances_val2017.json'  # Caminho para o arquivo de anotações original de validação
original_images_dir = 'data/val2017'  # Diretório original das imagens de validação

filtered_annotations_path = 'data/annotations/instances_val2017_filtered.json'  # Caminho para o novo arquivo de anotações de validação
filtered_images_dir = 'data/val2017_filtered'  # Diretório para as imagens filtradas de validação

# Lista das classes de interesse
classes_of_interest = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'dog'
]

# --- Carregar as anotações originais ---
print("Carregando as anotações validação originais...")
with open(original_annotations_path, 'r') as f:
    coco = json.load(f)

# --- Mapear nomes das classes para IDs ---
print("Mapeando nomes das classes para IDs...")
category_name_to_id = {category['name']: category['id'] for category in coco['categories']}
category_id_to_name = {category['id']: category['name'] for category in coco['categories']}

# Filtrar as categorias de interesse
category_ids_of_interest = [category_name_to_id[name] for name in classes_of_interest if name in category_name_to_id]

# --- Filtrar categorias ---
print("Filtrando categorias...")
filtered_categories = [category for category in coco['categories'] if category['id'] in category_ids_of_interest]

# --- Filtrar anotações ---
print("Filtrando anotações...")
filtered_annotations = [annotation for annotation in coco['annotations'] if annotation['category_id'] in category_ids_of_interest]

# --- Obter IDs das imagens que contêm as categorias de interesse ---
print("Obtendo IDs das imagens filtradas...")
image_ids = set(annotation['image_id'] for annotation in filtered_annotations)

# --- Filtrar imagens ---
print("Filtrando imagens...")
filtered_images = [image for image in coco['images'] if image['id'] in image_ids]

# --- Atualizar os IDs das categorias para começar em 0 ---
print("Atualizando IDs das categorias para começar em 0...")
old_category_id_to_new = {}
for new_id, category in enumerate(filtered_categories):
    old_id = category['id']
    old_category_id_to_new[old_id] = new_id  # Começa em 0
    category['id'] = new_id

# Atualizar 'category_id' nas anotações
for annotation in filtered_annotations:
    old_id = annotation['category_id']
    annotation['category_id'] = old_category_id_to_new[old_id]

# --- Criar o novo conjunto de anotações ---
print("Criando o novo arquivo de anotações...")
filtered_coco = {
    'info': coco['info'],
    'licenses': coco['licenses'],
    'images': filtered_images,
    'annotations': filtered_annotations,
    'categories': filtered_categories
}

with open(filtered_annotations_path, 'w') as f:
    json.dump(filtered_coco, f)

print(f"Novo arquivo de anotações salvo em: {filtered_annotations_path}")

# --- Copiar as imagens filtradas para um novo diretório ---
print("Copiando imagens filtradas para o novo diretório...")
if not os.path.exists(filtered_images_dir):
    os.makedirs(filtered_images_dir)

for image in tqdm(filtered_images, desc="Copiando imagens"):
    image_filename = image['file_name']
    src = os.path.join(original_images_dir, image_filename)
    dst = os.path.join(filtered_images_dir, image_filename)
    if os.path.exists(src):
        shutil.copyfile(src, dst)
    else:
        print(f"A imagem {image_filename} não foi encontrada no diretório original.")

print(f"Imagens copiadas para: {filtered_images_dir}")

print("Processo concluído com sucesso!")
print(f"Total de imagens filtradas: {len(filtered_images)}")
print(f"Total de anotações filtradas: {len(filtered_annotations)}")