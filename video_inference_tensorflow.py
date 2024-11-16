import os
import cv2
import numpy as np
import tensorflow as tf
from transformers import DetrImageProcessor
import math

# Configurações
CONFIDENCE_THRESHOLD = 0.2
MODEL_PATH = "detr_model_tf"  # Caminho do modelo TensorFlow
EXCLUSION_ZONE_HEIGHT_RATIO = 0.3  # Excluir 30% superior da imagem

# Multiplicador de perigo adicional para objetos dentro do trapézio
TRAPEZOID_DANGER_MULTIPLIER = 2

# Definir cores para cada nível de risco (em formato BGR)
risk_colors = {
    'muito_alto': (0, 0, 255),    # Vermelho
    'alto': (0, 165, 255),        # Laranja
    'medio': (0, 255, 255),       # Amarelo
    'baixo': (0, 255, 0),         # Verde
    'nenhum': (255, 255, 255)     # Branco
}

# Carregar o modelo TensorFlow
model = tf.saved_model.load(MODEL_PATH)

# Carregar o image_processor
image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

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
video_path = 'inference/jorge10.MP4'
cap = cv2.VideoCapture(video_path)

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
    inputs = image_processor(images=image, return_tensors="tf")

    # Redimensionar os tensores para corresponder às dimensões do modelo ONNX
    expected_height, expected_width = 1000, 1087  # Ajustar para o modelo exportado
    pixel_values = tf.image.resize(inputs["pixel_values"], [expected_height, expected_width])
    pixel_mask = tf.image.resize(tf.cast(inputs["pixel_mask"], tf.float32), [expected_height, expected_width])
    pixel_mask = tf.cast(pixel_mask, tf.int64)

    print("Pixel Values Shape:", pixel_values.shape)
    print("Pixel Mask Shape:", pixel_mask.shape)

    # Inferência com TensorFlow
    outputs = model.signatures["serving_default"](
        pixel_values=pixel_values,
        pixel_mask=pixel_mask
    )

    # Verificar as saídas do modelo
    logits = outputs["logits"].numpy()
    pred_boxes = outputs["pred_boxes"].numpy()
    print("Logits Shape:", logits.shape)
    print("Pred Boxes Shape:", pred_boxes.shape)

    # Pós-processamento
    boxes = pred_boxes[0]
    scores = logits[0]
    labels = np.argmax(scores, axis=-1)
    confidences = np.max(scores, axis=-1)

    # Filtrar detecções com base no threshold de confiança
    mask = confidences > CONFIDENCE_THRESHOLD
    boxes = boxes[mask]
    labels = labels[mask]
    confidences = confidences[mask]

    # Converter coordenadas dos boxes para o formato da imagem
    boxes[:, [0, 2]] *= image_width
    boxes[:, [1, 3]] *= image_height

    # Variável para acumular o risco total
    total_risk = 0
    warning_message = ""  # Variável para armazenar mensagem de alerta

    # Definir as coordenadas do trapézio de detecção
    top_left = (int(image_width * 0.45), exclusion_zone_height)
    top_right = (int(image_width * 0.50), exclusion_zone_height)
    bottom_left = (int(image_width * 0.10), image_height)
    bottom_right = (int(image_width * 0.90), image_height)

    # Avaliar as áreas de risco e calcular o risco total
    for bbox, label_id, confidence in zip(boxes, labels, confidences):
        label = id2label.get(label_id, f"Unknown({label_id})")
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
            # Aplicar o multiplicador de perigo adicional
            object_risk = danger_weight * proximity_weight * TRAPEZOID_DANGER_MULTIPLIER
        else:
            # Risco sem multiplicador
            object_risk = danger_weight * proximity_weight

        # Acumular o risco total
        total_risk += object_risk

        # Determinar o nível de risco individual
        if object_risk >= 40:
            individual_risk_level = 'muito_alto'
        elif object_risk >= 10:
            individual_risk_level = 'alto'
        elif object_risk >= 3:
            individual_risk_level = 'medio'
        elif object_risk >= 1:
            individual_risk_level = 'baixo'
        else:
            individual_risk_level = 'nenhum'

        # Obter a cor correspondente ao nível de risco
        box_color = risk_colors.get(individual_risk_level, (255, 255, 255))  # Branco como padrão

        # Desenhar o box no frame
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Classificar a cena com base no risco total
    if total_risk >= 60:
        risk_level = 'PARE'
    elif total_risk >= 40:
        risk_level = 'ACONSELHAVEL DIMINUIR A VELOCIDADE'
    elif total_risk >= 20:
        risk_level = 'DIMINUA A VELOCIDADE'
    else:
        risk_level = 'SEGURO'

    # Exibir o nível de risco no frame
    cv2.putText(frame, f"Risco: {risk_level} ({int(total_risk)})", (10, 70),
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