import json
import os
from pycocotools.coco import COCO

# Defina os caminhos para os arquivos
ANNOTATION_FILE = 'data/annotations/instances_train2017.json'
OUTPUT_ANNOTATION_FILE = 'data/annotations/instances_train2017_filtered.json'

# Lista de classes de interesse
classes_of_interest = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter'
]

# Carrega as anotações
with open(ANNOTATION_FILE, 'r') as f:
    coco_data = json.load(f)

# Mapeamento de nome para categoria ID (IDs originais)
category_name_to_id = {cat['name']: cat['id'] for cat in coco_data['categories']}

# IDs das categorias de interesse (IDs originais)
category_ids_of_interest = [category_name_to_id[class_name] for class_name in classes_of_interest]

print("IDs originais das categorias de interesse:", category_ids_of_interest)

# Criar mapeamento de IDs originais para IDs sequenciais iniciando em 0
id_map = {original_id: idx for idx, original_id in enumerate(category_ids_of_interest)}
print("Mapeamento de IDs originais para novos IDs:", id_map)

# Atualizar as categorias com os novos IDs
filtered_categories = []
for class_name in classes_of_interest:
    original_id = category_name_to_id[class_name]
    new_id = id_map[original_id]
    category = {
        'supercategory': class_name,
        'id': new_id,
        'name': class_name
    }
    filtered_categories.append(category)

print("Categorias filtradas com novos IDs:", filtered_categories)

# Filtrar e atualizar as anotações
filtered_annotations = []
for ann in coco_data['annotations']:
    original_category_id = ann['category_id']
    if original_category_id in category_ids_of_interest:
        # Atualizar o category_id para o novo ID
        ann['category_id'] = id_map[original_category_id]
        filtered_annotations.append(ann)

# Obter os IDs das imagens correspondentes
image_ids = set([ann['image_id'] for ann in filtered_annotations])

print(f"Número de anotações filtradas: {len(filtered_annotations)}")
print(f"Número de imagens correspondentes: {len(image_ids)}")

# Filtrar as imagens
filtered_images = [img for img in coco_data['images'] if img['id'] in image_ids]

print(f"Número de imagens filtradas: {len(filtered_images)}")

# Criar o novo dicionário de anotações
filtered_coco_data = {
    'info': coco_data.get('info', {}),
    'licenses': coco_data.get('licenses', []),
    'images': filtered_images,
    'annotations': filtered_annotations,
    'categories': filtered_categories
}

# Salvar o novo arquivo de anotações
with open(OUTPUT_ANNOTATION_FILE, 'w') as f:
    json.dump(filtered_coco_data, f)

print(f"Novo arquivo de anotações salvo em {OUTPUT_ANNOTATION_FILE}")
