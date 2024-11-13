import os
import cv2
import torch
import supervision as sv
import numpy as np
from transformers import DetrForObjectDetection, DetrImageProcessor
import math
import torchvision.ops as ops  # Import NMS from torchvision

# Configurações
CONFIDENCE_THRESHOLD = 0.2
CHECKPOINT = "facebook/detr-resnet-50"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NMS_IOU_THRESHOLD = 0.4  # Defina um limiar de IOU para NMS
EXCLUSION_ZONE_HEIGHT_RATIO = 0.3  # Excluir 30% superior da imagem

# Multiplicador de perigo adicional para objetos dentro do trapézio
TRAPEZOID_DANGER_MULTIPLIER = 2

# Carregar o modelo treinado
model = torch.load('outputs/detr_model_complete.pt')
model.to(DEVICE)
model.eval()

# Carregar o image_processor
image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

# Definir o mapeamento de IDs para labels
id2label = {
    0: 'Person-Mony-Bus-Tramway-Car-Tree',
    1: 'Bicycle',
    2: 'Bus',
    3: 'Car',
    4: 'Dog',
    5: 'Electric pole',
    6: 'Motorcycle',
    7: 'Person',
    8: 'Traffic signs',
    9: 'Tree',
    10: 'Uncovered manhole'
}

# Definir os níveis de perigo para cada classe
danger_levels = {
    'Person-Mony-Bus-Tramway-Car-Tree': 'nenhum',
    'Uncovered manhole': 'muito_alto',
    'Person': 'muito_alto',
    'Bicycle': 'medio',
    'Bus': 'medio',
    'Car': 'medio',
    'Dog': 'alto',
    'Motorcycle': 'medio',
    'Traffic signs': 'nenhum',
    'Electric pole': 'nenhum',
    'Tree': 'nenhum'
}

# Definir os pesos para os níveis de perigo
danger_weights = {
    'muito_alto': 40,
    'alto': 10,
    'medio': 3,
    'baixo': 1,
    'nenhum': 0,
}

# Função para obter o peso de proximidade com ajuste exponencial
def get_proximity_weight_exp(normalized_area):
    return int(5 * math.exp(-5 * (1 - normalized_area))) + 1

# Configurar a captura de vídeo
video_path = 'inference/jorge6.MP4'
cap = cv2.VideoCapture(video_path)

box_annotator = sv.BoxAnnotator()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar o frame para 720p
    new_width = 1280
    new_height = 720
    frame = cv2.resize(frame, (new_width, new_height))

    # Adicionar zona de exclusão superior com transparência
    exclusion_zone_height = int(EXCLUSION_ZONE_HEIGHT_RATIO * new_height)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (new_width, exclusion_zone_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)  # Transparência escura

    # Obter as dimensões da imagem
    image_height, image_width = frame.shape[:2]

    # Converter o frame para RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pré-processar a imagem
    inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)

    # Inferência
    with torch.no_grad():
        outputs = model(**inputs)

    # Pós-processamento
    target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
    results = image_processor.post_process_object_detection(
        outputs=outputs,
        threshold=CONFIDENCE_THRESHOLD,
        target_sizes=target_sizes
    )[0]

    # Aplicar NMS nas detecções
    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]

    # Executa NMS e recupera os índices das detecções a serem mantidas
    keep = ops.nms(boxes, scores, NMS_IOU_THRESHOLD)

    # Filtra as detecções com base nos índices mantidos
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Criar o objeto de detecções
    from supervision.detection.core import Detections
    detections = Detections(
        xyxy=boxes.cpu().numpy(),
        confidence=scores.cpu().numpy(),
        class_id=labels.cpu().numpy()
    )

    # Gerar as labels
    labels_text = [
        f"{id2label[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    # Variável para acumular o risco total
    total_risk = 0
    warning_message = ""  # Variável para armazenar mensagem de alerta

    # Definir as coordenadas do trapézio de detecção com bordas superiores mais fechadas
    top_left = (int(image_width * 0.45), exclusion_zone_height)
    top_right = (int(image_width * 0.50), exclusion_zone_height)
    bottom_left = (int(image_width * 0.10), image_height)
    bottom_right = (int(image_width * 0.90), image_height)

    # Avaliar as áreas de risco e calcular o risco total
    for bbox, class_id in zip(detections.xyxy, detections.class_id):
        label = id2label[class_id]
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        # Verificar se o objeto é um "Uncovered manhole"
        if label == "Uncovered manhole":
            warning_message = "CUIDADO, BURACO NA PISTA A FRENTE"

        # Calcular a área normalizada do bounding box
        bbox_area = (x2 - x1) * (y2 - y1)
        image_area = image_width * image_height
        normalized_area = bbox_area / image_area

        # Obter o peso de proximidade usando a função exponencial
        proximity_weight = get_proximity_weight_exp(normalized_area)

        # Obter os pesos do nível de perigo
        danger_level = danger_levels.get(label, 'baixo')
        danger_weight = danger_weights.get(danger_level, 1)

        # Verificar se o objeto está dentro do trapézio de detecção
        if (x_center > top_left[0] and x_center < top_right[0] and
                y_center > top_left[1] and y_center < bottom_left[1]):
            # Aplicar o multiplicador de perigo adicional para objetos dentro do trapézio
            object_risk = danger_weight * proximity_weight * TRAPEZOID_DANGER_MULTIPLIER
        else:
            # Risco sem multiplicador para objetos fora do trapézio
            object_risk = danger_weight * proximity_weight

        # Acumular o risco total
        total_risk += object_risk

    # Classificar a cena com base no risco total atualizado
    if total_risk >= 60:
        risk_level = 'PARE'
    elif total_risk >= 40:
        risk_level = 'ACONSELHAVEL DIMINUIR A VELOCIDADE'
    elif total_risk >= 20:
        risk_level = 'DIMINUA A VELOCIDADE'
    else:
        risk_level = 'SEGURO'

    # Anotar o frame
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels_text)

    # Desenhar o trapézio de detecção na imagem
    cv2.line(frame, top_left, bottom_left, (0, 255, 0), 2)
    cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
    cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
    cv2.line(frame, bottom_left, bottom_right, (0, 255, 0), 2)

    # Exibir o nível de risco no frame
    cv2.putText(frame, f"Risco: {risk_level} ({total_risk})", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Exibir mensagem de alerta de buraco, se detectado
    if warning_message:
        cv2.putText(frame, warning_message, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Exibir o frame
    cv2.imshow('Video Detection and Classification', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()