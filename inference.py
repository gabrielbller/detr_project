import os
import random
import cv2
import torch
import supervision as sv
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from transformers import DetrForObjectDetection, DetrImageProcessor

# Defina o limiar de confiança para as detecções
CONFIDENCE_TRESHOLD = 0.9
# Configurações atualizadas
CHECKPOINT = "facebook/detr-resnet-50"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Ajuste o caminho do dataset
dataset = os.path.join("datasets", "Obstacle-detection-11")

ANNOTATION_FILE_NAME = "_annotations.coco.json"
TEST_DIRECTORY = os.path.join(dataset, "test")

# Carrega o modelo e o processador de imagens
model = DetrForObjectDetection.from_pretrained('outputs')
model.to(DEVICE)

image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        image_directory_path: str,
        image_processor,
        train: bool = True
    ):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target

# Inicializar o dataset de teste
TEST_DATASET = CocoDetection(
    image_directory_path=TEST_DIRECTORY,
    image_processor=image_processor,
    train=False)

# Utils
categories = TEST_DATASET.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}
box_annotator = sv.BoxAnnotator()

# Selecionar uma imagem aleatória
image_ids = TEST_DATASET.coco.getImgIds()
image_id = random.choice(image_ids)
print('Image #{}'.format(image_id))

# Carregar a imagem e as anotações
image_info = TEST_DATASET.coco.loadImgs(image_id)[0]
annotations = TEST_DATASET.coco.imgToAnns[image_id]
image_path = os.path.join(TEST_DATASET.root, image_info['file_name'])
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converter para RGB

# Anotar a imagem com as anotações reais (ground truth)
detections_gt = sv.Detections.from_coco_annotations(coco_annotation=annotations)
labels_gt = [f"{id2label[class_id]}" for _, _, class_id, _ in detections_gt]
frame_gt = box_annotator.annotate(scene=image.copy(), detections=detections_gt, labels=labels_gt)

# Inferência com o modelo treinado
with torch.no_grad():
    inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)
    outputs = model(**inputs)

    # Pós-processamento
    target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
    results = image_processor.post_process_object_detection(
        outputs=outputs,
        threshold=CONFIDENCE_TRESHOLD,
        target_sizes=target_sizes
    )[0]

# Detecções do modelo
detections_pred = sv.Detections.from_transformers(transformers_results=results)
labels_pred = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections_pred]
frame_pred = box_annotator.annotate(scene=image.copy(), detections=detections_pred, labels=labels_pred)

# Mostrar as duas imagens lado a lado
fig, axes = plt.subplots(1, 2, figsize=(16, 16))

# Ground Truth
axes[0].imshow(frame_gt)
axes[0].set_title('Ground Truth')
axes[0].axis('off')

# Detections do Modelo
axes[1].imshow(frame_pred)
axes[1].set_title('Detections (Model)')
axes[1].axis('off')

# Exibir as imagens
plt.show()
