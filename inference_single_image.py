import tensorflow as tf
import cv2
import numpy as np
import time
import psutil
import math

# Parâmetros do modelo
MODEL_PATH = "detr_tf_model"
LABELS = ['Person-Mony-Bus-Tramway-Car-Tree', 'Bicycle', 'Bus', 'Car', 'Dog', 'Electric pole', 'Motorcycle', 'Person', 'Traffic signs', 'Tree', 'Uncovered manhole']
CONFIDENCE_THRESHOLD = 0.2

# Parâmetros de pré-processamento
IMAGE_SIZE = (720, 1280)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

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

# Ler a imagem de teste
IMAGE_PATH = "test_image.jpg"  # Substitua pelo caminho da sua imagem
image = cv2.imread(IMAGE_PATH)

# Métricas de desempenho
preprocess_start_time = time.time()
start_memory = psutil.Process(os.getpid()).memory_info().rss

# Pré-processar a imagem
input_tensor, original_shape = preprocess_image(image)
pixel_mask = np.ones((1, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=np.float32)
preprocess_end_time = time.time()
preprocess_time = preprocess_end_time - preprocess_start_time
preprocess_memory = psutil.Process(os.getpid()).memory_info().rss - start_memory

# Inferência
inference_start_time = time.time()
outputs = infer(pixel_values=tf.constant(input_tensor), pixel_mask=tf.constant(pixel_mask))
inference_end_time = time.time()
inference_time = inference_end_time - inference_start_time
inference_memory = psutil.Process(os.getpid()).memory_info().rss - start_memory

# Pós-processamento
postprocess_start_time = time.time()
boxes, scores, classes = postprocess_outputs(outputs, original_shape)
postprocess_end_time = time.time()
postprocess_time = postprocess_end_time - postprocess_start_time
postprocess_memory = psutil.Process(os.getpid()).memory_info().rss - start_memory

# Simular GT para métricas (substitua por ground truth real)
gt_boxes = np.array([[50, 50, 200, 200]])  # Substitua pelos GT reais
gt_classes = np.array([1])

# Calcular métricas de detecção
TP, FP, FN = 0, 0, 0
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

precision = TP / (TP + FP) if TP + FP > 0 else 0
recall = TP / (TP + FN) if TP + FN > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

# Calcular desempenho total
total_time = preprocess_time + inference_time + postprocess_time
total_memory = preprocess_memory + inference_memory + postprocess_memory
fps = 1 / total_time if total_time > 0 else 0

# Exibir métricas
print("\nMétricas gerais de detecção:")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")
print(f"Acurácia: {accuracy:.4f}")

print("\nMétricas de desempenho:")
print(f"Tempo de pré-processamento: {preprocess_time:.4f} s")
print(f"Tempo de inferência: {inference_time:.4f} s")
print(f"Tempo de pós-processamento: {postprocess_time:.4f} s")
print(f"Tempo total de execução: {total_time:.4f} s")
print(f"FPS: {fps:.2f}")

print("\nConsumo de memória:")
print(f"Memória para pré-processamento: {preprocess_memory / (1024 ** 2):.2f} MB")
print(f"Memória para inferência: {inference_memory / (1024 ** 2):.2f} MB")
print(f"Memória para pós-processamento: {postprocess_memory / (1024 ** 2):.2f} MB")
print(f"Memória total consumida: {total_memory / (1024 ** 2):.2f} MB")