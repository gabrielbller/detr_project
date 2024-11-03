import cv2
import torch
import supervision as sv
import numpy as np
import time  # Importar o módulo time
from transformers import DetrForObjectDetection, DetrImageProcessor

# Configurações
CONFIDENCE_THRESHOLD = 0.5
CHECKPOINT = "facebook/detr-resnet-50"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

# Configurar a captura de vídeo da câmera
cap = cv2.VideoCapture(0)

# Definir a resolução da câmera (opcional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

box_annotator = sv.BoxAnnotator()

frame_count = 0
process_every_n_frames = 2  # Processar a cada 2 frames

# Variáveis para cálculo do FPS
prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar frame da câmera. Verifique se a câmera está conectada e acessível.")
        break

        # Atualizar o tempo atual
    new_frame_time = time.time()

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

        # Anotar o frame
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # Calcular o FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

        # Atualizar o tempo atual
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Converter o FPS para string e formatar
    fps_text = f"FPS: {fps:.2f}"

    # Exibir o FPS no frame
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Exibir o frame
    cv2.imshow('Detecção em Tempo Real', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()