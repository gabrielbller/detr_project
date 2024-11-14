import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from tqdm import tqdm
import time
import numpy as np

# Funções auxiliares para calcular IoU e métricas
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_metrics(pred_boxes, pred_labels, pred_scores, true_boxes, true_labels, iou_threshold=0.5, score_threshold=0.5):
    TP, FP, FN = 0, 0, 0
    for i, pred_box in enumerate(pred_boxes):
        if pred_scores[i] < score_threshold:
            continue
        match_found = False
        for j, true_box in enumerate(true_boxes):
            if pred_labels[i] == true_labels[j] and calculate_iou(pred_box, true_box) >= iou_threshold:
                TP += 1
                match_found = True
                break
        if not match_found:
            FP += 1
    FN = len(true_boxes) - TP
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

# Função de avaliação para calcular métricas e FPS
@torch.no_grad()
def evaluate_model(model, data_loader, device, iou_threshold=0.5, score_threshold=0.5):
    model.eval()
    precisions, recalls, f1_scores = [], [], []
    total_time = 0
    total_images = 0

    for images, targets in tqdm(data_loader, desc="Avaliando"):
        images = [image.to(device) for image in images]
        
        # Tempo de início da inferência
        start_time = time.time()
        
        # Fazer inferência
        outputs = model(images)
        
        # Tempo de inferência
        inference_time = time.time() - start_time
        total_time += inference_time
        total_images += len(images)

        # Iterar sobre as previsões e os alvos correspondentes
        for output, target in zip(outputs, targets):
            pred_boxes = output['boxes'].cpu()
            pred_labels = output['labels'].cpu()
            pred_scores = output['scores'].cpu()

            # Verificar se `target` contém as chaves necessárias
            if len(target["boxes"]) == 0:
                continue  # Ignorar imagens sem caixas verdadeiras

            true_boxes = target["boxes"].cpu()
            true_labels = target["labels"].cpu()

            # Calcular precisão, recall e F1-score para cada imagem
            precision, recall, f1 = calculate_metrics(
                pred_boxes, pred_labels, pred_scores, true_boxes, true_labels,
                iou_threshold=iou_threshold, score_threshold=score_threshold
            )

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

    # Calcular FPS
    fps = total_images / total_time if total_time > 0 else 0
    print(f"Average Precision: {sum(precisions) / len(precisions):.4f}")
    print(f"Average Recall: {sum(recalls) / len(recalls):.4f}")
    print(f"Average F1-score: {sum(f1_scores) / len(f1_scores):.4f}")
    print(f"FPS (Frames per Second): {fps:.2f}")

# Carregar o modelo treinado
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = fasterrcnn_mobilenet_v3_large_fpn(weights=None).to(device)
num_classes = 11  # Número de classes incluindo o background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Carregar o modelo treinado
checkpoint = torch.load("output/fasterrcnn_mobilenet_epoch_1.pth", map_location=device)
model.load_state_dict(checkpoint)

# Configurar o dataset de validação
val_dataset_path = 'datasets/Obstacle-detection-11/valid'
val_ann_file = 'datasets/Obstacle-detection-11/valid/_annotations.coco.json'
transform = T.Compose([T.ToTensor()])
val_dataset = CocoDetection(root=val_dataset_path, annFile=val_ann_file, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Executa a avaliação
evaluate_model(model, val_loader, device, iou_threshold=0.5, score_threshold=0.5)
