from ultralytics import YOLO

def main():
    # Caminho absoluto para o modelo YOLOv8 pré-treinado que você já possui
    model_path = '/Users/gabri/Downloads/yolov8n.pt'  # Ajuste para o caminho completo do modelo

    # Carregar o modelo YOLOv8 a partir do caminho local
    model = YOLO(model_path)

    # Caminho para o arquivo de configuração do dataset no formato YOLO
    dataset_yaml = 'yolo/Obstacle-detection-11/data.yaml'

    # Parâmetros de treinamento
    epochs = 5  # número de épocas para o treinamento
    batch = 16   # tamanho do lote
    imgsz = 640  # tamanho da imagem para treino

    # Treinamento
    model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=0  # Use GPU se disponível; ou 'cpu' para CPU
    )

    # Avaliação do modelo após o fine-tuning
    metrics = model.val(data=dataset_yaml)
    print(metrics)  # Exibe as métricas de avaliação no conjunto de validação

# Protege o código principal
if __name__ == '__main__':
    main()
