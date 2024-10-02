import os

# Diretórios dos conjuntos de dados filtrados
train_image_dir = 'data/train2017_filtered'
val_image_dir = 'data/val2017_filtered'

# Contar o número de arquivos no diretório de treinamento
train_image_count = len(os.listdir(train_image_dir))
print(f"Número de imagens no dataset de treinamento: {train_image_count}")

# Contar o número de arquivos no diretório de validação
val_image_count = len(os.listdir(val_image_dir))
print(f"Número de imagens no dataset de validação: {val_image_count}")
