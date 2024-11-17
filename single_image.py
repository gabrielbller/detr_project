import tensorflow as tf
import cv2
import numpy as np

# Configurações do modelo e parâmetros
MODEL_PATH = "detr_tf_model"
IMAGE_PATH = "path/to/your/image.jpg"  # Substitua pelo caminho da sua imagem
LABELS = ['Person-Mony-Bus-Tramway-Car-Tree', 'Bicycle', 'Bus', 'Car', 'Dog', 'Electric pole', 'Motorcycle', 'Person', 'Traffic signs', 'Tree', 'Uncovered manhole']
CONFIDENCE_THRESHOLD = 0.2

# Configurações de pré-processamento
IMAGE_SIZE = (720, 1280)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

# Função para pré-processar a imagem
def preprocess_image(image):
    image_resized = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_normalized = (image_rgb / 255.0 - MEAN) / STD
    image_expanded = np.expand_dims(image_normalized, axis=0).astype(np.float32)
    return image_expanded, image_resized.shape[:2]

# Função para pós-processar as saídas do modelo
def postprocess_outputs(outputs, original_shape):
    boxes = outputs['detection_boxes'].numpy()
    scores = outputs['detection_scores'].numpy()
    classes = outputs['detection_classes'].numpy()

    # Converter boxes para coordenadas absolutas
    h, w = original_shape
    boxes[:, [0, 2]] *= h  # Escalar alturas
    boxes[:, [1, 3]] *= w  # Escalar larguras

    # Filtrar por limiar de confiança
    keep = scores >= CONFIDENCE_THRESHOLD
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]
    return boxes, scores, classes

# Função para desenhar as detecções na imagem
def draw_detections(image, boxes, scores, classes):
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{LABELS[int(cls)]}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Carregar o modelo
print("Carregando o modelo...")
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]

# Ler e processar a imagem
image = cv2.imread(IMAGE_PATH)
input_tensor, original_shape = preprocess_image(image)

# Criar a máscara de pixels válidos
pixel_mask = np.ones((1, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=np.float32)

# Realizar a inferência
print("Realizando a inferência...")
outputs = infer(pixel_values=tf.constant(input_tensor), pixel_mask=tf.constant(pixel_mask))

# Pós-processar as saídas
boxes, scores, classes = postprocess_outputs(outputs, original_shape)

# Desenhar as detecções na imagem
image_with_detections = draw_detections(image, boxes, scores, classes)

# Exibir a imagem com as detecções
cv2.imshow("Detections", image_with_detections)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Salvar a imagem com as detecções
output_path = "inference/output_image.jpg"
cv2.imwrite(output_path, image_with_detections)
print(f"Imagem salva em: {output_path}")