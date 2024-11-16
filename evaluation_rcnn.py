import os
import time
import numpy as np
import torch
import psutil
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Configurações
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CONFIDENCE_THRESHOLD = 0.9  # Ajuste conforme necessário
IOU_THRESHOLD = 0.2  # Threshold de IoU para considerar uma detecção como correta

dataset = os.path.join("datasets", "Obstacle-detection-11")
ANNOTATION_FILE_NAME = "_annotations.coco.json"
TEST_DIRECTORY = os.path.join(dataset, "test")

# **Definir o número de classes**
NUM_CLASSES = 11  # Substitua pelo número correto de classes (incluindo o background)

# **Inicializar o modelo Faster R-CNN**
model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=NUM_CLASSES)

# **Carregar o state_dict**
model.load_state_dict(torch.load('faster_rcnn.pth', map_location=DEVICE))

# **Mover o modelo para o dispositivo e definir para modo de avaliação**
model.to(DEVICE)
model.eval()

# **Definir transformações personalizadas**

class ToTensor:
    def __call__(self, image, target):
        if not isinstance(image, torch.Tensor):  # Verificar se a imagem já é um tensor
            image = transforms.functional.to_tensor(image)
        return image, target

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

# Criar a transformação
transform = Compose([
    ToTensor(),
    # Adicione outras transformações se necessário
])

# Para armazenar tempos e memória de cada etapa
preprocess_times = []
inference_times = []
postprocess_times = []
preprocess_memory = []
inference_memory = []
postprocess_memory = []

def collate_fn(batch):
    images = list(item[0] for item in batch)
    targets = list(item[1] for item in batch)
    return images, targets

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, image_directory_path: str, transforms=None):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(
            image_directory_path,
            annotation_file_path,
            transforms=None,  # Não passar as transformações para a classe base
            transform=None,
            target_transform=None
        )
        self.transforms = transforms  # Definir as transformações na classe personalizada
    
    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
    
        # Converter anotações COCO para o formato esperado pelo Faster R-CNN
        boxes = []
        labels = []
        for annotation in target['annotations']:
            xmin = annotation['bbox'][0]
            ymin = annotation['bbox'][1]
            width = annotation['bbox'][2]
            height = annotation['bbox'][3]
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(annotation['category_id'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
    
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([image_id])
    
        if self.transforms is not None:
            img, target = self.transforms(img, target)
    
        return img, target

# Dataset e DataLoader
TEST_DATASET = CocoDetection(
    image_directory_path=TEST_DIRECTORY,
    transforms=transform
)
TEST_DATALOADER = torch.utils.data.DataLoader(
    dataset=TEST_DATASET,
    collate_fn=collate_fn,
    batch_size=1  # Usando batch_size=1 para simplificar
)

# Inicializar contadores e métricas
TP = 0
FP = 0
FN = 0

# Inicializar a métrica de mAP
metric_map = MeanAveragePrecision(iou_thresholds=[IOU_THRESHOLD])

print("Executando avaliação...")

for batch_idx, (images, targets) in enumerate(tqdm(TEST_DATALOADER)):
    start_memory = psutil.Process(os.getpid()).memory_info().rss  # Memória no início (em bytes)
    
    # Pré-processamento
    preprocess_start_time = time.time()
    images = list(image.to(DEVICE) for image in images)
    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
    preprocess_end_time = time.time()
    preprocess_times.append(preprocess_end_time - preprocess_start_time)
    preprocess_memory.append(psutil.Process(os.getpid()).memory_info().rss - start_memory)
    
    # Inferência
    inference_start_time = time.time()
    with torch.no_grad():
        outputs = model(images)
    inference_end_time = time.time()
    inference_times.append(inference_end_time - inference_start_time)
    inference_memory.append(psutil.Process(os.getpid()).memory_info().rss - start_memory)

    # Pós-processamento (no caso do Faster R-CNN, não é necessário)
    postprocess_start_time = time.time()
    postprocess_end_time = time.time()
    postprocess_times.append(postprocess_end_time - postprocess_start_time)
    postprocess_memory.append(psutil.Process(os.getpid()).memory_info().rss - start_memory)

    for target, output in zip(targets, outputs):
        gt_boxes = target['boxes']
        gt_labels = target['labels']

        pred_boxes = output['boxes']
        pred_scores = output['scores']
        pred_labels = output['labels']

        # Filtrar predições pelo limiar de confiança
        keep = pred_scores >= CONFIDENCE_THRESHOLD
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        # Preparar entradas para a métrica mAP
        preds = [
            dict(
                boxes=pred_boxes,
                scores=pred_scores,
                labels=pred_labels
            )
        ]
        targets_metric = [
            dict(
                boxes=gt_boxes,
                labels=gt_labels
            )
        ]

        # Atualizar a métrica mAP
        metric_map.update(preds, targets_metric)

        # Listas para marcar correspondências
        matched_gt = set()
        matched_pred = set()

        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            # Calcular IoUs
            ious = box_iou(pred_boxes, gt_boxes)

            # Obter índices das predições e anotações ordenados pelos IoUs
            iou_values, indices = ious.flatten().sort(descending=True)
            pred_indices, gt_indices = indices // gt_boxes.size(0), indices % gt_boxes.size(0)

            for idx in range(len(iou_values)):
                if iou_values[idx] < IOU_THRESHOLD:
                    break

                pred_idx = pred_indices[idx].item()
                gt_idx = gt_indices[idx].item()

                if pred_idx in matched_pred or gt_idx in matched_gt:
                    continue

                pred_label = pred_labels[pred_idx].item()
                gt_label = gt_labels[gt_idx].item()

                if pred_label == gt_label:
                    TP += 1
                    matched_pred.add(pred_idx)
                    matched_gt.add(gt_idx)
                else:
                    FP += 1
                    matched_pred.add(pred_idx)

            # Falsos Positivos: predições não correspondidas
            num_unmatched_preds = len(pred_boxes) - len(matched_pred)
            FP += num_unmatched_preds

            # Falsos Negativos: anotações não correspondidas
            num_unmatched_gts = len(gt_boxes) - len(matched_gt)
            FN += num_unmatched_gts
        else:
            # Se não houver predições, todos os gt_boxes são FN
            FN += len(gt_boxes)
            # Se não houver gt_boxes, todas as predições são FP
            FP += len(pred_boxes)

# Calcular métricas
precision = TP / (TP + FP) if TP + FP > 0 else 0
recall = TP / (TP + FN) if TP + FN > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
accuracy = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0  # Adicionando cálculo da acurácia


# Obter as métricas de AP e mAP
map_metrics = metric_map.compute()
average_precision = map_metrics['map']

# Cálculo de tempos e uso de memória
total_preprocess_time = sum(preprocess_times)
total_inference_time = sum(inference_times)
total_postprocess_time = sum(postprocess_times)
total_time = total_preprocess_time + total_inference_time + total_postprocess_time

avg_preprocess_memory = np.mean(preprocess_memory) / (1024 ** 2)
avg_inference_memory = np.mean(inference_memory) / (1024 ** 2)
avg_postprocess_memory = np.mean(postprocess_memory) / (1024 ** 2)
total_memory = avg_preprocess_memory + avg_inference_memory + avg_postprocess_memory

avg_fps = 1 / np.mean(inference_times)

print("\nMétricas gerais de detecção:")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")
print(f"Acurácia: {accuracy:.4f}")  # Exibindo a acurácia
print(f"Precisão Média (mAP) @ IoU={IOU_THRESHOLD}: {average_precision:.4f}")

print("\nMétricas de desempenho:")
print(f"Tempo total de pré-processamento: {total_preprocess_time:.4f} s")
print(f"Tempo total de inferência: {total_inference_time:.4f} s")
print(f"Tempo total de pós-processamento: {total_postprocess_time:.4f} s")
print(f"Tempo total de execução: {total_time:.4f} s")
print(f"FPS médio: {avg_fps:.2f}")

print("\nConsumo médio de memória:")
print(f"Média de memória para pré-processamento: {avg_preprocess_memory:.2f} MB")
print(f"Média de memória para inferência: {avg_inference_memory:.2f} MB")
print(f"Média de memória para pós-processamento: {avg_postprocess_memory:.2f} MB")
print(f"Memória total consumida: {total_memory:.2f} MB")