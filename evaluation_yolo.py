from ultralytics import YOLO
import time

def main():
    def calcular_acuracia_e_fps(model_path, dataset_yaml):
        # Carregar o modelo YOLOv8
        modelo = YOLO(model_path)

        # Inicializar lista para armazenar tempos de inferência
        tempos_inferencia = []

        # Avaliar o modelo no conjunto de dados de validação e medir o tempo
        resultados = modelo.val(data=dataset_yaml, save_json=True, verbose=False)

        # Realiza a inferência e mede o tempo para cada imagem no conjunto de validação
        for batch in resultados.data_loader:
            inicio = time.time()
            modelo.predict(batch[0])  # Executa a inferência em uma imagem
            fim = time.time()
            tempos_inferencia.append(fim - inicio)

        # Calcula o tempo médio por imagem e o FPS médio
        tempo_medio_inferencia = sum(tempos_inferencia) / len(tempos_inferencia) if tempos_inferencia else 0
        fps_medio = 1 / tempo_medio_inferencia if tempo_medio_inferencia > 0 else 0

        # Obter métricas de precisão e recall
        precision_media = resultados.metrics['precision']
        recall_media = resultados.metrics['recall']
        acuracia = (precision_media + recall_media) / 2 if (precision_media + recall_media) > 0 else 0

        # Exibir os resultados
        print("\nMétricas de Desempenho do Modelo:")
        print(f"Precisão Média (Mean Precision): {precision_media:.4f}")
        print(f"Recall Médio (Mean Recall): {recall_media:.4f}")
        print(f"Acurácia Aproximada: {acuracia:.4f}")
        print(f"FPS Médio: {fps_medio:.2f}")

    # Caminho para o modelo YOLOv8 e o dataset de validação
    model_path = 'runs/detect/train3/weights/best.pt'  # Substitua pelo caminho para o seu modelo YOLOv8
    dataset_yaml = 'yolo/Obstacle-detection-11/data.yaml'  # Substitua pelo caminho para o seu conjunto de dados de validação

    # Executa a função para calcular a acurácia e o FPS médio
    calcular_acuracia_e_fps(model_path, dataset_yaml)

if __name__ == '__main__':
    main()
