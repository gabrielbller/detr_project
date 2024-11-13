import os
import cv2
import torch
import supervision as sv
import numpy as np
from transformers import DetrImageProcessor
from collections import deque

# Configurações
CONFIDENCE_THRESHOLD = 0.2
CHECKPOINT = "facebook/detr-resnet-50"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Parâmetros da janela temporal e limiares de risco
TEMPORAL_WINDOW = 30  # quantidade de frames para acumular dados (~1 segundo a 30 FPS)
HIGH_RISK_THRESHOLD = 80
MODERATE_RISK_THRESHOLD = 50

# Histórico de risco para acumular riscos de cada frame
risk_history = deque(maxlen=TEMPORAL_WINDOW)

# Carregar o modelo treinado
model = torch.load('outputs/detr_model_complete.pt', map_location=DEVICE)
model.eval()

# Carregar o image_processor
image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

# Mapear IDs para labels
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
    'Uncovered manhole': 'alto',
    'Person': 'muito_alto',
    'Bicycle': 'medio',
    'Bus': 'medio',
    'Car': 'medio',
    'Dog': 'medio',
    'Motorcycle': 'medio',
    'Traffic signs': 'medio',
    'Electric pole': 'nenhum',
    'Tree': 'nenhum'
}

# Definir os pesos para os níveis de perigo
danger_weights = {
    'muito_alto': 5,
    'alto': 3,
    'medio': 2,
    'baixo': 1,
    'nenhum': 0
}

# Histórico de posições para calcular movimento
object_history = {}

def calculate_movement_weight(object_id, bbox, image_center):
    if object_id in object_history:
        last_bbox = object_history[object_id]
        x1, y1, x2, y2 = bbox
        prev_x1, prev_y1, prev_x2, prev_y2 = last_bbox
        movement = np.sqrt((x1 - prev_x1)**2 + (y1 - prev_y1)**2)
        dist_to_center = np.sqrt(((x1 + x2) / 2 - image_center[0])**2 + ((y1 + y2) / 2 - image_center[1])**2)
        if movement > 5 and dist_to_center < image_center[0] * 0.5:
            return 1.5
    object_history[object_id] = bbox
    return 1

def calculate_risk_for_frame(detections, image_center, image_width, image_height):
    total_risk = 0
    for i, (bbox, class_id, score) in enumerate(zip(detections.xyxy, detections.class_id, detections.confidence)):
        label = id2label[class_id]
        x1, y1, x2, y2 = bbox
        bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        bbox_area = (x2 - x1) * (y2 - y1)
        image_area = image_width * image_height
        normalized_area = bbox_area / image_area

        danger_weight = danger_weights.get(danger_levels.get(label, 'baixo'), 1)

        movement_weight = calculate_movement_weight(i, bbox, image_center)
        distance = np.sqrt((bbox_center[0] - image_center[0])**2 + (bbox_center[1] - image_center[1])**2)
        proximity_weight = 1 if distance > image_width * 0.5 else 2 if distance > image_width * 0.25 else 3

        object_risk = proximity_weight * danger_weight * movement_weight
        total_risk += object_risk
    
    return total_risk

def classify_scene_risk():
    if len(risk_history) < TEMPORAL_WINDOW:
        return "Calculando..."  # Ainda não há dados suficientes para classificar
    
    average_risk = np.mean(risk_history)
    
    if average_risk >= HIGH_RISK_THRESHOLD:
        return 'ACIDENTE IMINENTE'
    elif average_risk >= MODERATE_RISK_THRESHOLD:
        return 'RISCO MODERADO - REDUZIR VELOCIDADE'
    else:
        return 'SEGURO'

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if not ret:
        print(f"Não foi possível ler o vídeo {video_path}")
        return

    image_height, image_width = frame.shape[:2]
    image_center = (image_width / 2, image_height / 2)
    new_width, new_height = 1280, 720

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (new_width, new_height))

    while ret:
        frame = cv2.resize(frame, (new_width, new_height))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
        results = image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=CONFIDENCE_THRESHOLD,
            target_sizes=target_sizes
        )[0]

        detections = sv.Detections(
            xyxy=results["boxes"].cpu().numpy(),
            confidence=results["scores"].cpu().numpy(),
            class_id=results["labels"].cpu().numpy()
        )

        frame_risk = calculate_risk_for_frame(detections, image_center, image_width, image_height)
        risk_history.append(frame_risk)
        scene_risk = classify_scene_risk()

        for bbox, class_id, score in zip(detections.xyxy, detections.class_id, detections.confidence):
            x1, y1, x2, y2 = bbox.astype(int)
            label_text = f"{id2label[class_id]} {score:.2f}"
            
            box_color = (0, 255, 0) if scene_risk == 'SEGURO' else (0, 165, 255) if scene_risk == 'RISCO MODERADO - REDUZIR VELOCIDADE' else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), box_color, -1)
            cv2.putText(frame, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, f"Risco: {scene_risk}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)
        ret, frame = cap.read()

    cap.release()
    out.release()

# Pasta de entrada e saída
input_folder = 'inference'
output_folder = 'output_videos'
os.makedirs(output_folder, exist_ok=True)

# Processar todos os vídeos na pasta de entrada
for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        print(f"Processando vídeo: {filename}")
        process_video(input_path, output_path)

print("Processamento de vídeos concluído.")
