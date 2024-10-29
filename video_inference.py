import os
import cv2
import torch
import supervision as sv
import numpy as np
from transformers import DetrForObjectDetection, DetrImageProcessor

# Configurações
CONFIDENCE_THRESHOLD = 0.6
CHECKPOINT = "facebook/detr-resnet-50"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Carregar o modelo treinado
model = DetrForObjectDetection.from_pretrained('outputs')
model.to(DEVICE)

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
    'Person-Mony-Bus-Tramway-Car-Tree': 'alto',
    'Uncovered manhole': 'alto',
    'Person': 'muito_alto',
    'Bicycle': 'medio',
    'Bus': 'medio',
    'Car': 'medio',
    'Dog': 'medio',
    'Motorcycle': 'medio',
    'Traffic signs': 'medio',
    'Electric pole': 'baixo',
    'Tree': 'baixo'
}

# Definir os pesos para os níveis de perigo
danger_weights = {
    'muito_alto': 5,
    'alto': 3,
    'medio': 2,
    'baixo': 1
}

# Definir os pesos para as zonas verticais
zone_weights = {
    'central': 3,   # Alta prioridade
    'lateral': 1    # Baixa prioridade
}

# Função para obter o peso de proximidade
def get_proximity_weight(normalized_area):
    if normalized_area > 0.05:
        return 3  # Muito próximo
    elif normalized_area > 0.02:
        return 2  # Próximo
    else:
        return 1  # Distante

# Configurar a captura de vídeo
video_path = 'inference/jorge/jorge1.MP4'
cap = cv2.VideoCapture(video_path)

box_annotator = sv.BoxAnnotator()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar o frame
    new_width = 720
    new_height = 480
    frame = cv2.resize(frame, (new_width, new_height))

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

    # Criar o objeto de detecções
    from supervision.detection.core import Detections
    detections = Detections(
        xyxy=results["boxes"].cpu().numpy(),
        confidence=results["scores"].cpu().numpy(),
        class_id=results["labels"].cpu().numpy()
    )

    # Gerar as labels
    labels = [
        f"{id2label[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    # Variável para acumular o risco total
    total_risk = 0

    # Definir os limites da zona central (aumentada)
    central_zone_left = image_width * 0.30
    central_zone_right = image_width * 0.70

    # Avaliar as áreas de risco e calcular o risco total
    for bbox, class_id in zip(detections.xyxy, detections.class_id):
        label = id2label[class_id]
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        # Verificar se o objeto está na zona central
        if central_zone_left <= x_center <= central_zone_right:
            zone = 'central'
        else:
            zone = 'lateral'

        # Considerar apenas objetos na zona central
        if zone == 'central':
            # Calcular a área normalizada do bounding box
            bbox_area = (x2 - x1) * (y2 - y1)
            image_area = image_width * image_height
            normalized_area = bbox_area / image_area

            # Obter o peso de proximidade
            proximity_weight = get_proximity_weight(normalized_area)

            # Obter os pesos da zona e do nível de perigo
            zone_weight = zone_weights.get(zone, 1)
            danger_level = danger_levels.get(label, 'baixo')
            danger_weight = danger_weights.get(danger_level, 1)

            # Calcular o risco do objeto
            object_risk = zone_weight * danger_weight * proximity_weight

            # Acumular o risco total
            total_risk += object_risk

    # Classificar a cena com base no risco total
    if total_risk >= 60:
        risk_level = 'STOP'
    elif total_risk >= 45:
        risk_level = 'WARNING DANGER AHEAD'
    elif total_risk >= 20:
        risk_level = 'SLOW DOWN'
    else:
        risk_level = 'SAFE'

    # Anotar o frame
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # Desenhar as zonas verticais na imagem
    # Zona central (linhas verdes)
    cv2.line(frame, (int(central_zone_left), 0), (int(central_zone_left), image_height), (0, 255, 0), 2)
    cv2.line(frame, (int(central_zone_right), 0), (int(central_zone_right), image_height), (0, 255, 0), 2)

    # Opcional: preencher as zonas laterais com transparência
    overlay = frame.copy()
    alpha = 0.2  # Transparência

    # Preencher as zonas laterais (amarelo)
    cv2.rectangle(overlay, (0, 0), (int(central_zone_left), image_height), (1, 1, 1), -1)  # Lateral esquerda
    cv2.rectangle(overlay, (int(central_zone_right), 0), (image_width, image_height), (1, 1, 1), -1)  # Lateral direita


    # Combinar o overlay com o frame original
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Exibir o nível de risco no frame
    cv2.putText(frame, f"Risco: {risk_level} ({total_risk})", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Exibir o frame
    cv2.imshow('Video Detection and Classification', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
