import os
import random
import numpy as np
import torch
import cv2
import torchvision
import supervision as sv
import matplotlib.pyplot as plt
from transformers import DetrImageProcessor
from pycocotools.coco import COCO
from train import CocoDetection, Detr


# Configurações atualizadas
CHECKPOINT = "facebook/detr-resnet-50"
ANNOTATION_FILE_NAME = "data/annotations/instances_train2017_filtered.json"
TRAIN_DIRECTORY = "data/train2017_filtered"
VAL_ANNOTATION_FILE_NAME = "data/annotations/instances_val2017_filtered.json"
VAL_DIRECTORY = "data/val2017_filtered"
TEST_ANNOTATION_FILE_NAME = "data/annotations/instances_val2017_filtered.json"
TEST_DIRECTORY = "data/val2017_filtered"


image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

# Defina o dispositivo (CPU ou GPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Usando o dispositivo: {DEVICE}')

# Defina o limiar de confiança para as detecções
CONFIDENCE_TRESHOLD = 0.5

# # Carregar as anotações filtradas
# coco = COCO(ANNOTATION_FILE_NAME)
# categories = coco.loadCats(coco.getCatIds())

# # Criar mapeamentos
# id2label = {0: 'N/A'}  # Adicionar a classe 'N/A' com ID 0
# id2label.update({idx+1: cat['name'] for idx, cat in enumerate(categories)})
# label2id = {cat['id']: idx+1 for idx, cat in enumerate(categories)}

# num_labels = len(id2label)

# print(f"Número de classes: {num_labels}")

# # Inicialize o image_processor (usado durante o treinamento)
# image_processor = DetrImageProcessor.from_pretrained(
#     CHECKPOINT,
#     size=480,
#     id2label=id2label,
#     label2id=label2id
# )

# Carregue o modelo treinado
CHECKPOINT_PATH = 'outputs/detr-epoch=09-validation_loss=0.00.ckpt'

model = Detr.load_from_checkpoint(
    CHECKPOINT_PATH,
    lr=1e-4,
    lr_backbone=1e-5,
    weight_decay=1e-4
)
# model.eval()
# model.to(DEVICE)

# Inicializar o dataset de teste
TEST_DATASET = CocoDetection(
    image_directory_path=TEST_DIRECTORY,
    annotation_file_path=TEST_ANNOTATION_FILE_NAME,
    image_processor=image_processor
)

# Utils
categories = TEST_DATASET.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}
box_annotator = sv.BoxAnnotator()

# Selecionar uma imagem aleatória
image_ids = TEST_DATASET.coco.getImgIds()
image_id = random.choice(image_ids)
print('Image #{}'.format(image_id))

# Carregar a imagem e as anotações
image = TEST_DATASET.coco.loadImgs(image_id)[0]
annotations = TEST_DATASET.coco.imgToAnns[image_id]
image_path = os.path.join(TEST_DATASET.root, image['file_name'])
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converter para RGB

# Anotar a imagem com as anotações reais (ground truth)
detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)
labels = [f"{id2label[class_id]}" for _, _, class_id, _ in detections]
frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

# Exibir a imagem com as anotações reais
print('Anotações Reais (Ground Truth)')
plt.figure(figsize=(16, 16))
plt.imshow(frame)
plt.axis('off')
plt.show()

# Inferência com o modelo treinado
with torch.no_grad():
    # Pré-processar a imagem
    inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)
    outputs = model(**inputs)
    
    # Pós-processar as saídas do modelo
    target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
    results = image_processor.post_process_object_detection(
        outputs=outputs,
        threshold=CONFIDENCE_TRESHOLD,
        target_sizes=target_sizes
    )[0]

# Indexar as detecções com os índices selecionados
detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=0.5)
labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]
frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

print(f"Detecções encontradas: {len(results['scores'])}")
print("Scores:", results['scores'])
print("Labels:", results['labels'])
print("Boxes:", results['boxes'])


# Exibir a imagem com as detecções do modelo
print('Detecções do Modelo')
plt.figure(figsize=(16, 16))
plt.imshow(frame)
plt.axis('off')
plt.show()
