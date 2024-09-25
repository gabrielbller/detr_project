import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision import datasets, ops
from torchvision.models import resnet50, ResNet50_Weights
from torch.cuda.amp import autocast, GradScaler
from einops import rearrange
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import gc
import time
import copy

# Número de classes (12 classes de interesse + "no object")
num_classes = 12
num_classes_with_no_object = num_classes + 1

# Lista das classes de interesse
classes_of_interest = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'dog'
]

# Cores para visualização (opcional)
COLORS = [
    [0.000, 0.447, 0.741],  # azul
    [0.850, 0.325, 0.098],  # laranja
    [0.929, 0.694, 0.125],  # amarelo
    [0.494, 0.184, 0.556],  # roxo
    [0.466, 0.674, 0.188],  # verde
    [0.301, 0.745, 0.933],  # ciano
]
COLORS *= 100  # Repete as cores para ter o suficiente

# Função para pré-processar as anotações
def preprocess_target(anno, im_w, im_h):
    # Filtra anotações com 'iscrowd' == 0
    anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
    boxes = [obj["bbox"] for obj in anno]
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

    # Converte de xywh para xyxy
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=im_w)
    boxes[:, 1::2].clamp_(min=0, max=im_h)
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]

    classes = [obj["category_id"] for obj in anno]
    classes = torch.tensor(classes, dtype=torch.int64)
    classes = classes[keep]

    # Subtrai 1 dos IDs das categorias para que comecem em 0 (já que o category_id começa em 1)
    classes -= 1

    # Verifica se as categorias estão no intervalo [0, num_classes - 1]
    if not torch.all((classes >= 0) & (classes < num_classes)):
        raise ValueError(f"Category IDs fora do intervalo permitido [0, {num_classes - 1}].")

    # Normaliza boxes para [0, 1]
    boxes[:, 0::2] /= im_w
    boxes[:, 1::2] /= im_h
    boxes.clamp_(min=0, max=1)

    boxes = ops.box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh')
    target = {}
    target['labels'] = classes
    target['boxes'] = boxes
    return target

# Classe personalizada para o dataset COCO com data augmentation
class MyCocoDetection(datasets.CocoDetection):
    def __init__(self, *args, transforms=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        w, h = img.size
        target = preprocess_target(target, w, h)
        target['image_id'] = torch.tensor(self.ids[idx])
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

# Transformações de data augmentation
class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, target):
        if np.random.rand() < self.prob:
            img = T.functional.hflip(img)
            w, h = img.size
            bbox = target['boxes']
            bbox[:, 0] = 1 - bbox[:, 0]
            target['boxes'] = bbox
        return img, target

class ToTensor:
    def __call__(self, img, target):
        img = T.ToTensor()(img)
        return img, target

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        img = T.Resize(self.size, antialias=True)(img)
        return img, target

class Normalize:
    def __init__(self, mean, std):
        self.normalize = T.Normalize(mean, std)

    def __call__(self, img, target):
        img = self.normalize(img)
        return img, target

# Instancia o dataset e o dataloader com data augmentation
transform = ComposeTransforms([
    RandomHorizontalFlip(),
    ToTensor(),
    Resize((480, 480)),
    Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
])

dataset = MyCocoDetection(
    'data/train2017_filtered',
    'data/annotations/instances_train2017_filtered.json',
    transforms=transform
)

# Divide o dataset em treinamento e validação
torch.manual_seed(42)
indices = torch.randperm(len(dataset)).tolist()
train_size = int(0.9 * len(indices))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_ds = torch.utils.data.Subset(dataset, train_indices)
val_ds = torch.utils.data.Subset(dataset, val_indices)

print(f'\nNúmero de samples de treinamento: {len(train_ds)}')
print(f'Número de samples de validação: {len(val_ds)}')

# Função para agrupar batches
def collate_fn(inputs):
    input_ = [i[0] for i in inputs]
    targets = [i[1] for i in inputs]
    return input_, targets

# Definição do modelo DETR
class DETR(nn.Module):
    def __init__(self, num_classes=num_classes, num_queries=100):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        hidden_dim = 256
        self.conv1x1 = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
        )

        self.positional_encoding = nn.Parameter(torch.rand(50 * 50, hidden_dim))
        self.query_pos = nn.Parameter(torch.rand(num_queries, hidden_dim))

        self.class_embed = nn.Linear(hidden_dim, num_classes_with_no_object)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, x):
        x = self.backbone(x)
        h = self.conv1x1(x)
        bs, c, h_w, w_w = h.shape

        h_flat = h.flatten(2).permute(2, 0, 1)  # [H*W, bs, hidden_dim]

        pos = self.positional_encoding[:h_w * w_w, :].unsqueeze(1).repeat(1, bs, 1)

        query_pos = self.query_pos.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_pos)

        memory = self.transformer.encoder(h_flat + pos)
        hs = self.transformer.decoder(tgt + query_pos, memory)

        outputs_class = self.class_embed(hs.transpose(0, 1))
        outputs_coord = self.bbox_embed(hs.transpose(0, 1)).sigmoid()

        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}

class MLP(nn.Module):
    """ Multi-layer Perceptron """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Função de correspondência
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_logits'].shape[:2]

        # Flatten para calcular as matrizes de custo em lote
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs['pred_boxes'].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Concatena os labels e boxes alvo
        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets])

        # Calcula o custo de classificação
        cost_class = -out_prob[:, tgt_ids]

        # Calcula o custo L1 entre boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Calcula o custo GIoU entre boxes
        cost_giou = -ops.generalized_box_iou(
            ops.box_convert(out_bbox, 'cxcywh', 'xyxy'),
            ops.box_convert(tgt_bbox, 'cxcywh', 'xyxy')
        )

        # Matriz de custo final
        C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v['boxes']) for v in targets]
        indices = []
        for i, c in enumerate(C):
            c = c[:, :sizes[i]]
            indices.append(linear_sum_assignment(c))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

# Função para calcular a perda
class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, eos_coef):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef

        weight = torch.ones(num_classes + 1)
        weight[-1] = self.eos_coef
        self.register_buffer('weight', weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']  # [batch_size, num_queries, num_classes]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.weight)
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        src_boxes = outputs['pred_boxes']

        idx = self._get_src_permutation_idx(indices)
        src_boxes = src_boxes[idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(ops.generalized_box_iou(
            ops.box_convert(src_boxes, 'cxcywh', 'xyxy'),
            ops.box_convert(target_boxes, 'cxcywh', 'xyxy')
        )).sum() / num_boxes

        losses = {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

        return losses

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t['boxes']) for t in targets)

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

# Função para calcular o mAP
def evaluate(model, data_loader, device):
    model.eval()
    coco_gt = COCO('data/annotations/instances_val2017_filtered.json')  # Arquivo de anotações do conjunto de validação
    coco_dt = []
    img_ids = []
    with torch.no_grad():
        for imgs, targets in data_loader:
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)
            outputs = {k: v.cpu() for k, v in outputs.items()}
            for img, target, pred_logits, pred_boxes in zip(imgs, targets, outputs['pred_logits'], outputs['pred_boxes']):
                # Obtém o ID da imagem
                image_id = target['image_id'].item()
                img_ids.append(image_id)

                # Processa as predições
                probas = pred_logits.softmax(-1)[..., :-1]  # Exclui a classe "no object"
                scores, labels = probas.max(-1)

                # Filtra as detecções com base em um threshold de confiança
                keep = scores > 0.05  # Você pode ajustar este valor
                scores = scores[keep]
                labels = labels[keep]
                boxes = pred_boxes[keep]

                # Converte as caixas para o formato COCO (xywh)
                boxes = ops.box_convert(boxes, 'cxcywh', 'xywh').numpy()
                boxes[:, :2] -= boxes[:, 2:] / 2  # Ajusta as coordenadas

                # Multiplica pelas dimensões da imagem para obter coordenadas absolutas
                img_w, img_h = img.shape[2], img.shape[1]
                scale_fct = np.array([img_w, img_h, img_w, img_h])
                boxes = boxes * scale_fct

                for score, label, box in zip(scores.numpy(), labels.numpy(), boxes):
                    coco_dt.append({
                        "image_id": int(image_id),
                        "category_id": int(label + 1),  # Adiciona 1 para corresponder ao COCO
                        "bbox": box.tolist(),
                        "score": float(score)
                    })
    # Salva as predições em um arquivo JSON temporário
    with open('temp_predictions.json', 'w') as f:
        json.dump(coco_dt, f, indent=4)

    # Carrega as predições usando o COCO
    coco_dt = coco_gt.loadRes('temp_predictions.json')

    # Cria o objeto COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = img_ids  # Avalia apenas as imagens do conjunto de validação

    # Executa a avaliação
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Remove o arquivo temporário
    import os
    os.remove('temp_predictions.json')

    model.train()

# Treinamento do Modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detr = DETR(num_classes=num_classes, num_queries=100).to(device)

# Definir a função de correspondência e a função de perda
matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
criterion = SetCriterion(num_classes, matcher=matcher, eos_coef=0.1).to(device)

# Definir o otimizador com taxas de aprendizado diferentes
param_dicts = [
    {"params": [p for n, p in detr.named_parameters() if "backbone" not in n and p.requires_grad]},
    {"params": [p for n, p in detr.named_parameters() if "backbone" in n and p.requires_grad], "lr": 1e-5},
]
optimizer = AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)

# Scheduler de taxa de aprendizado
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Configurações de treinamento
batch_size = 16  # Aumentado o tamanho do batch
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
num_epochs = 50
print_every_n = 100
best_model_wts = copy.deepcopy(detr.state_dict())
best_loss = float('inf')

detr.train()
scaler = GradScaler()  # Para treinamento de precisão mista

for epoch in range(num_epochs):
    start_time = time.time()
    losses = []
    for i, (input_, targets) in enumerate(train_loader):
        input_ = [img.to(device) for img in input_]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast():
            outputs = detr(input_)
            loss_dict = criterion(outputs, targets)
            weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
            losses_total = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        scaler.scale(losses_total).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(losses_total.item())

        # Limpeza de memória
        torch.cuda.empty_cache()
        gc.collect()

        if (i + 1) % print_every_n == 0:
            loss_avg = np.mean(losses[-print_every_n:])
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss_avg:.4f}')

    # Avaliação no conjunto de validação
        val_loss = 0.0
    detr.eval()
    with torch.no_grad():
        for input_, targets in val_loader:
            input_ = [img.to(device) for img in input_]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = detr(input_)
            loss_dict = criterion(outputs, targets)
            weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
            losses_total = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            val_loss += losses_total.item()
    val_loss /= len(val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
    scheduler.step(val_loss)

    # Chama a função evaluate para calcular o mAP
    evaluate(detr, val_loader, device)

    # Salva o melhor modelo
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(detr.state_dict())
        torch.save({
            'model_state_dict': best_model_wts,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }, f'{outputs}/best_checkpoint.pt')
        print('Model saved!')

    detr.train()
    epoch_time = time.time() - start_time
    print(f'Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s')

# Carrega os pesos do melhor modelo
detr.load_state_dict(best_model_wts)

# Função para visualizar as predições
def visualize_predictions(detr, dataloader):
    detr.eval()
    device = next(detr.parameters()).device  # Obtém o dispositivo (CPU ou GPU)
    for input_, targets in dataloader:
        input_ = [img.to(device) for img in input_]
        with torch.no_grad():
            outputs = detr(input_)
            for idx in range(len(input_)):
                probas = outputs['pred_logits'][idx].softmax(-1)[:, :-1]
                keep = probas.max(-1).values > 0.7

                bboxes_scaled = ops.box_convert(outputs['pred_boxes'][idx][keep].cpu(), 'cxcywh', 'xyxy')
                plt.figure(figsize=(16, 10))
                img = input_[idx].cpu().permute(1, 2, 0).numpy()
                # Desnormaliza a imagem
                img = img * np.array([0.229, 0.224, 0.225])
                img = img + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                plt.imshow(img)
                ax = plt.gca()
                colors = COLORS * 100  # Garante que há cores suficientes
                for p, (xmin, ymin, xmax, ymax), c in zip(probas[keep], bboxes_scaled.tolist(), colors):
                    cl = p.argmax()
                    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                               fill=False, color=c, linewidth=3))
                    text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
                    ax.text(xmin, ymin, text, fontsize=15,
                            bbox=dict(facecolor='yellow', alpha=0.5))
                plt.axis('off')
                plt.show()
        break  # Apenas um batch para visualização
    detr.train()
