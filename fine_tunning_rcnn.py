import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
from tqdm import tqdm  # Biblioteca para visualização de progresso

# Configurações
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4
NUM_CLASSES = 11  # Número de classes + 1 (classe de fundo)
EPOCHS = 1
LEARNING_RATE = 0.005
DATASET_DIR = "datasets/Obstacle-detection-11"
ANNOTATION_FILE = "_annotations.coco.json"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "valid")

# Função de transformação (aumentos e normalização)
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset e DataLoader
train_dataset = CocoDetection(
    root=TRAIN_DIR,
    annFile=os.path.join(TRAIN_DIR, ANNOTATION_FILE),
    transform=transform
)
val_dataset = CocoDetection(
    root=VAL_DIR,
    annFile=os.path.join(VAL_DIR, ANNOTATION_FILE),
    transform=transform
)

def collate_fn(batch):
    images, targets = tuple(zip(*batch))
    processed_images = []
    processed_targets = []
    for img, t in zip(images, targets):
        boxes = [obj['bbox'] for obj in t if 'bbox' in obj and obj['bbox']]
        if len(boxes) == 0:
            # Ignorar imagens sem bounding boxes
            continue
        boxes = torch.tensor(
            [[x, y, x + w, y + h] for x, y, w, h in boxes],
            dtype=torch.float32
        )
        labels = torch.tensor([obj['category_id'] for obj in t if 'bbox' in obj and obj['bbox']], dtype=torch.int64)
        processed_images.append(img)
        processed_targets.append({
            "boxes": boxes,
            "labels": labels
        })
    return processed_images, processed_targets

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Verificar o formato do dataset
print("Verificando o formato do dataset...")
image, target = train_dataset[0]
print("Imagem:", type(image), image.shape)
print("Target:", target)

# Carregar modelo pré-treinado
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Modificar o cabeçalho do modelo para o número de classes do dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

model.to(DEVICE)

# Otimizador e Scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Função de treinamento com barra de progresso
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Treinando Época {epoch+1}")
    for images, targets in progress_bar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=losses.item())
    print(f"Perda Média na Época {epoch+1}: {total_loss / len(data_loader):.4f}")

# Função de validação com barra de progresso
def evaluate(model, data_loader, device, epoch):
    model.eval()
    progress_bar = tqdm(data_loader, desc=f"Avaliando Época {epoch+1}")
    with torch.no_grad():
        for images, targets in progress_bar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            model(images)

# Treinamento
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
    evaluate(model, val_loader, DEVICE, epoch)
    lr_scheduler.step()

# Salvar o modelo
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "faster_rcnn.pth"))
print("Modelo salvo!")