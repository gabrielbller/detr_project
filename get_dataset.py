import os
from roboflow import Roboflow

ROBOFLOW_API_KEY = input('Enter ROBOFLOW_API_KEY secret value: ')
# api key = uqTXlACGPeIn0KqpFHb7

# Define o diretório atual e cria a pasta 'datasets' dentro dele
current_directory = os.getcwd()
datasets_directory = os.path.join(current_directory, 'datasets')
os.makedirs(datasets_directory, exist_ok=True)

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("visually-impaired-obstacle-detection-uxdze").project("obstacle-detection-yeuzf")
version = project.version(11)
dataset = version.download("coco")

print(f'Dataset salvo em: {dataset.location}')
