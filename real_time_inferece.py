import cv2
import torch
import supervision as sv
import numpy as np
from transformers import DetrForObjectDetection, DetrImageProcessor

# Configurações
CONFIDENCE_THRESHOLD = 0.5
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

# Configurar a captura de vídeo da câmera
cap = cv2.VideoCapture(0)

# Definir a resolução da câmera (opcional)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

box_annotator = sv.BoxAnnotator()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar frame da câmera. Verifique se a câmera está conectada e acessível.")
        break

    # Redimensionar o frame (opcional)
    # new_width = 1080
    # new_height = 920
    # frame = cv2.resize(frame, (new_width, new_height))

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

    # Exibir o frame
    cv2.imshow('Detecção em Tempo Real', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
