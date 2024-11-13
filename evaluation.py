import os
import time
import numpy as np
import torch
import psutil 
import torchvision
from torchvision.ops import box_iou
from tqdm import tqdm
from transformers import DetrImageProcessor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Configurações
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = 'facebook/detr-resnet-50'
CONFIDENCE_THRESHOLD = 0.9
IOU_THRESHOLD = 0.1

dataset = os.path.join("datasets", "Obstacle-detection-11")
ANNOTATION_FILE_NAME = "_annotations.coco.json"
TEST_DIRECTORY = os.path.join(dataset, "test")

# Carregar modelo DETR
model = torch.load('outputs/detr_model_complete.pt', map_location=DEVICE)
model.to(DEVICE)
model.eval()

image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

# Para armazenar tempos e memória de cada etapa
preprocess_times = []
inference_times = []
postprocess_times = []
preprocess_memory = []
inference_memory = []
postprocess_memory = []

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, image_directory_path: str, image_processor):
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
        target['image_id'] = torch.tensor([image_id])
        return pixel_values, target

# Dataset e DataLoader
TEST_DATASET = CocoDetection(
    image_directory_path=TEST_DIRECTORY,
    image_processor=image_processor
)
TEST_DATALOADER = torch.utils.data.DataLoader(
    dataset=TEST_DATASET,
    collate_fn=collate_fn,
    batch_size=1  # Usando batch_size=1 para simplificar
)

# Função para converter caixas de [cx, cy, w, h] para [xmin, ymin, xmax, ymax]
def cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes.unbind(-1)
    x1 = x_c - 0.5 * w
    y1 = y_c - 0.5 * h
    x2 = x_c + 0.5 * w
    y2 = y_c + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

# Inicializar contadores e métricas
TP = 0
FP = 0
FN = 0

# Inicializar a métrica de mAP
metric_map = MeanAveragePrecision(iou_thresholds=[IOU_THRESHOLD])

print("Executando avaliação...")

for batch_idx, batch in enumerate(tqdm(TEST_DATALOADER)):
    start_memory = psutil.Process(os.getpid()).memory_info().rss  # Memória no início (em bytes)
    
    # Pré-processamento
    preprocess_start_time = time.time()
    pixel_values = batch["pixel_values"].to(DEVICE)
    pixel_mask = batch["pixel_mask"].to(DEVICE)
    labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]
    preprocess_end_time = time.time()
    preprocess_times.append(preprocess_end_time - preprocess_start_time)
    preprocess_memory.append(psutil.Process(os.getpid()).memory_info().rss - start_memory)
    
    # Inferência
    inference_start_time = time.time()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    inference_end_time = time.time()
    inference_times.append(inference_end_time - inference_start_time)
    inference_memory.append(psutil.Process(os.getpid()).memory_info().rss - start_memory)

    # Pós-processamento
    postprocess_start_time = time.time()
    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = image_processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes)
    postprocess_end_time = time.time()
    postprocess_times.append(postprocess_end_time - postprocess_start_time)
    postprocess_memory.append(psutil.Process(os.getpid()).memory_info().rss - start_memory)

    for target, result in zip(labels, results):
        gt_boxes = target['boxes']  # Formato [cx, cy, w, h]
        gt_labels = target['class_labels']

        pred_boxes = result['boxes']
        pred_scores = result['scores']
        pred_labels = result['labels']

        # Filtrar predições pelo limiar de confiança
        keep = pred_scores >= CONFIDENCE_THRESHOLD
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        # Converter caixas de ground truth para [xmin, ymin, xmax, ymax] e dimensionar para pixels absolutos
        gt_boxes = cxcywh_to_xyxy(gt_boxes)
        image_height, image_width = target['orig_size'].tolist()
        scaling_factors = torch.tensor([image_width, image_height, image_width, image_height]).to(DEVICE)
        gt_boxes = gt_boxes * scaling_factors

        # Preparar entradas para a métrica mAP
        preds = [
            dict(
                boxes=pred_boxes,
                scores=pred_scores,
                labels=pred_labels
            )
        ]
        targets = [
            dict(
                boxes=gt_boxes,
                labels=gt_labels
            )
        ]

        # Atualizar a métrica mAP
        metric_map.update(preds, targets)

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
