import os
import tensorflow as tf
import cv2
import numpy as np
import time
import psutil
from tqdm import tqdm

# Configurações do modelo
MODEL_PATH = "detr_tf_model"
LABELS = ['Person-Mony-Bus-Tramway-Car-Tree', 'Bicycle', 'Bus', 'Car', 'Dog', 'Electric pole', 'Motorcycle', 'Person', 'Traffic signs', 'Tree', 'Uncovered manhole']
CONFIDENCE_THRESHOLD = 0.2

# Configurações de pré-processamento
IMAGE_SIZE = (720, 1280)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

# Caminhos do dataset
dataset = os.path.join("datasets", "Obstacle-detection-11")
ANNOTATION_FILE_NAME = "_annotations.coco.json"
TEST_DIRECTORY = os.path.join(dataset, "test")

# Funções auxiliares
def preprocess_image(image):
    image_resized = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_normalized = (image_rgb / 255.0 - MEAN) / STD
    image_expanded = np.expand_dims(image_normalized, axis=0).astype(np.float32)
    return image_expanded, image_resized.shape[:2]

def postprocess_outputs(outputs, original_shape):
    boxes = outputs['detection_boxes'].numpy()
    scores = outputs['detection_scores'].numpy()
    classes = outputs['detection_classes'].numpy()

    # Converter boxes para coordenadas absolutas
    h, w = original_shape
    boxes[:, [0, 2]] *= h
    boxes[:, [1, 3]] *= w

    # Filtrar por limiar de confiança
    keep = scores >= CONFIDENCE_THRESHOLD
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]
    return boxes, scores, classes

# Função para calcular IoU
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

# Carregar o modelo
print("Carregando o modelo...")
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]

# Configurações para medição de desempenho
preprocess_times, inference_times, postprocess_times = [], [], []
preprocess_memory, inference_memory, postprocess_memory = [], [], []
TP, FP, FN = 0, 0, 0

# Obter lista de imagens no diretório de teste
image_paths = [os.path.join(TEST_DIRECTORY, f) for f in os.listdir(TEST_DIRECTORY) if f.endswith((".jpg", ".png"))]

print("Processando o conjunto de dados de teste...")
for image_path in tqdm(image_paths):
    start_memory = psutil.Process(os.getpid()).memory_info().rss

    # Ler a imagem
    image = cv2.imread(image_path)

    # Pré-processamento
    preprocess_start_time = time.time_ns()
    input_tensor, original_shape = preprocess_image(image)
    pixel_mask = np.ones((1, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=np.float32)  # Criar máscara de pixels válidos
    preprocess_end_time = time.time_ns()
    preprocess_times.append((preprocess_end_time - preprocess_start_time) / 1e9)
    preprocess_memory.append(psutil.Process(os.getpid()).memory_info().rss - start_memory)

    # Inferência
    inference_start_time = time.time_ns()
    outputs = infer(pixel_values=tf.constant(input_tensor), pixel_mask=tf.constant(pixel_mask))
    inference_end_time = time.time_ns()
    inference_times.append((inference_end_time - inference_start_time) / 1e9)
    inference_memory.append(psutil.Process(os.getpid()).memory_info().rss - start_memory)

    # Pós-processamento
    postprocess_start_time = time.time_ns()
    boxes, scores, classes = postprocess_outputs(outputs, original_shape)
    postprocess_end_time = time.time_ns()
    postprocess_times.append((postprocess_end_time - postprocess_start_time) / 1e9)
    postprocess_memory.append(psutil.Process(os.getpid()).memory_info().rss - start_memory)

    # Simulação de GT (substituir por dados reais)
    gt_boxes = np.array([[50, 50, 200, 200]])  # Substituir por ground truth real
    gt_classes = np.array([1])

    # Calcular métricas (substitua pela lógica real de correspondência GT-Pred)
    matched_pred, matched_gt = set(), set()
    for i, pred_box in enumerate(boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou >= 0.5 and classes[i] == gt_classes[j]:
                TP += 1
                matched_pred.add(i)
                matched_gt.add(j)

    FP += len(boxes) - len(matched_pred)
    FN += len(gt_boxes) - len(matched_gt)

# Calcular métricas gerais
precision = TP / (TP + FP) if TP + FP > 0 else 0
recall = TP / (TP + FN) if TP + FN > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0  # Cálculo da acurácia

# Calcular desempenho
total_preprocess_time = sum(preprocess_times)
total_inference_time = sum(inference_times)
total_postprocess_time = sum(postprocess_times)
total_time = total_preprocess_time + total_inference_time + total_postprocess_time
avg_fps = len(image_paths) / total_time

avg_preprocess_memory = np.mean(preprocess_memory) / (1024 ** 2)
avg_inference_memory = np.mean(inference_memory) / (1024 ** 2)
avg_postprocess_memory = np.mean(postprocess_memory) / (1024 ** 2)
total_memory = avg_preprocess_memory + avg_inference_memory + avg_postprocess_memory

# Exibir métricas
print("\nMétricas gerais de detecção:")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")
print(f"Acurácia: {accuracy:.4f}")  # Exibindo a acurácia

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