import tensorflow as tf
import cv2
import numpy as np
import math
import time  # Import necessário para medir o tempo

# Parâmetros do modelo
MODEL_PATH = "detr_tf_model"  # Caminho para o modelo TensorFlow salvo
LABELS = ['Person-Mony-Bus-Tramway-Car-Tree', 'Bicycle', 'Bus', 'Car', 'Dog', 'Electric pole', 'Motorcycle', 'Person', 'Traffic signs', 'Tree', 'Uncovered manhole']
NUM_CLASSES = len(LABELS)
CONFIDENCE_THRESHOLD = 0.2   # Limite de confiança para exibir detecções

# Parâmetros de pré-processamento
IMAGE_SIZE = (720, 1280)  # Ajustado para 720p (altura, largura)
MEAN = np.array([0.485, 0.456, 0.406])  # Média usada na normalização
STD = np.array([0.229, 0.224, 0.225])   # Desvio padrão usado na normalização

# Configurações adicionais
NMS_IOU_THRESHOLD = 0.4  # Limite de IOU para NMS
EXCLUSION_ZONE_HEIGHT_RATIO = 0.3  # Excluir 30% superior da imagem
TRAPEZOID_DANGER_MULTIPLIER = 2  # Multiplicador de perigo adicional para objetos dentro do trapézio

# Definir cores para cada nível de risco (em formato BGR)
risk_colors = {
    'muito_alto': (0, 0, 255),    # Vermelho
    'alto': (0, 165, 255),        # Laranja
    'medio': (0, 255, 255),       # Amarelo
    'baixo': (0, 255, 0),         # Verde
    'nenhum': (255, 255, 255)     # Branco
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

# Função para pré-processar as imagens
def preprocess_image(image):
    # Redimensionar a imagem para IMAGE_SIZE
    image_resized = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    
    # Converter BGR para RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    # Normalizar a imagem
    image_normalized = (image_rgb / 255.0 - MEAN) / STD
    
    # Transpor para (C, H, W)
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    
    # Expandir dimensões para (1, C, H, W)
    image_expanded = np.expand_dims(image_transposed, axis=0).astype(np.float32)
    
    return image_expanded, image_resized.shape[:2]  # Retorna também o tamanho do frame processado

# Função para pós-processar as saídas do modelo
def postprocess_outputs(outputs, original_image_shape, processed_image_shape):
    logits = outputs['logits'][0].numpy()        # (num_queries, num_classes)
    pred_boxes = outputs['pred_boxes'][0].numpy()  # (num_queries, 4)

    # Aplicar softmax nos logits para obter probabilidades
    probas = tf.nn.softmax(logits, axis=-1).numpy()  # (num_queries, num_classes)

    # Selecionar as classes com maior probabilidade
    max_probs = np.max(probas[:, :-1], axis=1)  # Exclui a classe 'no object' (última classe)
    max_classes = np.argmax(probas[:, :-1], axis=1)
    
    # Filtrar detecções com confiança abaixo do limite
    keep = max_probs > CONFIDENCE_THRESHOLD
    filtered_boxes = pred_boxes[keep]
    filtered_classes = max_classes[keep]
    filtered_scores = max_probs[keep]

    # Aplicar NMS
    if len(filtered_boxes) > 0:
        # Converter caixas para formato [y1, x1, y2, x2] para o NMS do TensorFlow
        boxes_tf = filtered_boxes.copy()
        boxes_tf[:, 0] = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2  # y_min
        boxes_tf[:, 1] = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2  # x_min
        boxes_tf[:, 2] = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2  # y_max
        boxes_tf[:, 3] = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2  # x_max

        selected_indices = tf.image.non_max_suppression(
            boxes=boxes_tf,
            scores=filtered_scores,
            max_output_size=100,
            iou_threshold=NMS_IOU_THRESHOLD
        ).numpy()
        
        filtered_boxes = filtered_boxes[selected_indices]
        filtered_classes = filtered_classes[selected_indices]
        filtered_scores = filtered_scores[selected_indices]
    else:
        filtered_boxes = np.array([])
        filtered_classes = np.array([])
        filtered_scores = np.array([])

    # Converter caixas para coordenadas da imagem original
    h_o, w_o = original_image_shape[:2]
    h_p, w_p = processed_image_shape  # Tamanho da imagem após o pré-processamento
    scale_x = w_o / w_p
    scale_y = h_o / h_p

    boxes_scaled = filtered_boxes * [w_p, h_p, w_p, h_p]
    boxes_scaled[:, [0, 2]] *= scale_x  # Ajuste no eixo x
    boxes_scaled[:, [1, 3]] *= scale_y  # Ajuste no eixo y

    # Converter de (cx, cy, w, h) para (x_min, y_min, x_max, y_max)
    boxes_converted = np.zeros_like(boxes_scaled)
    boxes_converted[:, 0] = boxes_scaled[:, 0] - boxes_scaled[:, 2] / 2  # x_min
    boxes_converted[:, 1] = boxes_scaled[:, 1] - boxes_scaled[:, 3] / 2  # y_min
    boxes_converted[:, 2] = boxes_scaled[:, 0] + boxes_scaled[:, 2] / 2  # x_max
    boxes_converted[:, 3] = boxes_scaled[:, 1] + boxes_scaled[:, 3] / 2  # y_max

    return boxes_converted, filtered_classes, filtered_scores

# Função para desenhar detecções, calcular riscos e exibir o FPS médio
def draw_detections_and_calculate_risk(image, boxes, classes, scores, frame_count, avg_fps):
    total_risk = 0
    warning_message = ""

    # Obter dimensões da imagem
    image_height, image_width = image.shape[:2]

    # Definir zona de exclusão
    exclusion_zone_height = int(EXCLUSION_ZONE_HEIGHT_RATIO * image_height)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image_width, exclusion_zone_height), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

    # Definir coordenadas do trapézio de detecção
    top_left = (int(image_width * 0.45), exclusion_zone_height)
    top_right = (int(image_width * 0.55), exclusion_zone_height)
    bottom_left = (int(image_width * 0.10), image_height)
    bottom_right = (int(image_width * 0.90), image_height)

    # Desenhar o trapézio de detecção na imagem
    cv2.line(image, top_left, bottom_left, (0, 255, 0), 2)
    cv2.line(image, top_right, bottom_right, (0, 255, 0), 2)
    cv2.line(image, top_left, top_right, (0, 255, 0), 2)
    cv2.line(image, bottom_left, bottom_right, (0, 255, 0), 2)

    # Lista para armazenar as cores das caixas
    box_colors = []

    for box, cls, score in zip(boxes, classes, scores):
        x_min, y_min, x_max, y_max = box.astype(int)
        label = LABELS[cls]
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        # Verificar se o objeto é um "Uncovered manhole"
        if label == "Uncovered manhole":
            warning_message = "CUIDADO, BURACO NA PISTA À FRENTE"

        # Calcular a área normalizada do bounding box
        bbox_area = (x_max - x_min) * (y_max - y_min)
        image_area = image_width * image_height
        normalized_area = bbox_area / image_area

        # Obter o peso de proximidade usando a função exponencial
        proximity_weight = get_proximity_weight_exp(normalized_area)

        # Obter os pesos do nível de perigo
        danger_level = danger_levels.get(label, 'baixo')
        danger_weight = danger_weights.get(danger_level, 1)

        # Verificar se o objeto está dentro do trapézio de detecção
        in_trapezoid = (
            x_center > top_left[0] and x_center < top_right[0] and
            y_center > top_left[1] and y_center < bottom_left[1]
        )

        if in_trapezoid:
            # Aplicar o multiplicador de perigo adicional para objetos dentro do trapézio
            object_risk = danger_weight * proximity_weight * TRAPEZOID_DANGER_MULTIPLIER
        else:
            # Risco sem multiplicador para objetos fora do trapézio
            object_risk = danger_weight * proximity_weight

        # Acumular o risco total
        total_risk += object_risk

        # Determinar o nível de risco individual com base no risco do objeto
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

        # Obter a cor correspondente ao nível de risco individual
        box_color = risk_colors.get(individual_risk_level, (255, 255, 255))  # Branco como padrão
        box_colors.append(box_color)

        # Desenhar retângulo
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, 2)

        # Colocar rótulo
        label_text = f"{label}: {score:.2f}"
        cv2.putText(image, label_text, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Classificar a cena com base no risco total
    if total_risk >= 60:
        risk_level = 'PARE'
    elif total_risk >= 40:
        risk_level = 'ACONSELHÁVEL DIMINUIR A VELOCIDADE'
    elif total_risk >= 20:
        risk_level = 'DIMINUA A VELOCIDADE'
    else:
        risk_level = 'SEGURO'

    # Exibir o nível de risco no frame
    cv2.putText(image, f"Risco: {risk_level} ({int(total_risk)})", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Exibir mensagem de alerta de buraco, se detectado
    if warning_message:
        cv2.putText(image, warning_message, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Exibir o FPS médio no canto superior direito
    fps_text = f"FPS Medio: {avg_fps:.2f}"
    text_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_x = image_width - text_size[0] - 10
    text_y = 30
    cv2.putText(image, fps_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return image

# Carregar o modelo TensorFlow
model = tf.saved_model.load(MODEL_PATH)

# Função de inferência
infer = model.signatures['serving_default']

# Abrir o vídeo
VIDEO_PATH = "inference/jorge10.MP4"  # Substitua pelo caminho do seu vídeo
cap = cv2.VideoCapture(VIDEO_PATH)

# Verificar se o vídeo foi aberto corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Definir o codec e criar o objeto VideoWriter para salvar o vídeo em 720p
out = cv2.VideoWriter('video_output_720p.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), 30,  # Você pode ajustar o FPS conforme necessário
                      (1280, 720))  # Resolução 720p

frame_count = 0  # Contador de frames
total_time = 0.0  # Tempo total de processamento

# Processar o vídeo quadro a quadro
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    start_time = time.time()  # Iniciar contagem de tempo

    # Redimensionar o frame para 720p
    frame_resized = cv2.resize(frame, (1280, 720))

    # Pré-processar o quadro
    input_tensor, processed_image_shape = preprocess_image(frame_resized)

    # Criar a máscara de pixel (todos os pixels são válidos)
    pixel_mask = np.ones((1, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=np.float32)

    # Executar inferência
    inputs = {
        'pixel_values': tf.constant(input_tensor),
        'pixel_mask': tf.constant(pixel_mask)
    }
    outputs = infer(**inputs)

    # Pós-processar as saídas
    boxes, classes, scores = postprocess_outputs(outputs, frame_resized.shape, processed_image_shape)

    # Calcular o tempo de processamento do frame
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_time += elapsed_time

    # Calcular o FPS médio
    avg_fps = frame_count / total_time

    # Desenhar detecções, calcular riscos e exibir FPS no quadro original
    frame_with_detections = draw_detections_and_calculate_risk(frame_resized, boxes, classes, scores, frame_count, avg_fps)

    # Exibir o quadro com detecções (opcional)
    cv2.imshow('Detections', frame_with_detections)

    # Escrever o quadro no vídeo de saída
    out.write(frame_with_detections)

    # Pressionar 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()