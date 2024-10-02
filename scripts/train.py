import torch
import torchvision
from transformers import DetrForObjectDetection, DetrImageProcessor
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


# Configurações atualizadas
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = "facebook/detr-resnet-50"
CONFIDENCE_TRESHOLD = 0.5
IOU_TRESHOLD = 0.8

ANNOTATION = "data/annotations/instances_train2017_filtered.json"
TRAIN_DIRECTORY = "data/train2017_filtered"
VAL_ANNOTATION = "data/annotations/instances_val2017_filtered.json"
VAL_DIRECTORY = "data/val2017_filtered"
TEST_ANNOTATION = "data/annotations/instances_val2017_filtered.json"
TEST_DIRECTORY = "data/val2017_filtered"

image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, image_directory_path: str, annotation_file_path: str, image_processor, train: bool = True):
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)        
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target
    
# Inicializar os datasets
TRAIN_DATASET = CocoDetection(
    image_directory_path=TRAIN_DIRECTORY,
    annotation_file_path=ANNOTATION,
    image_processor=image_processor,
    train=True
)

VAL_DATASET = CocoDetection(
    image_directory_path=VAL_DIRECTORY,
    annotation_file_path=VAL_ANNOTATION,
    image_processor=image_processor,
    train=False
)

TEST_DATASET = CocoDetection(
    image_directory_path=TEST_DIRECTORY,
    annotation_file_path=TEST_ANNOTATION, 
    image_processor=image_processor, 
    train=False
)

print("Number of training examples:", len(TRAIN_DATASET))
print("Number of validation examples:", len(VAL_DATASET))
print("Number of test examples:", len(TEST_DATASET))

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

# Inicialização dos DataLoaders
TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET,collate_fn=collate_fn,batch_size=2,shuffle=True,num_workers=0)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET,collate_fn=collate_fn,batch_size=2,num_workers=0)
TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=2, num_workers=0)

categories = TRAIN_DATASET.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}
print("id2label:", id2label)


class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT,
            num_labels=len(id2label),
            ignore_mismatched_sizes=True
        )

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        # Congela o backbone para acelerar o treinamento
        for param in self.model.model.backbone.parameters():
            param.requires_grad = False

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())
            
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return TRAIN_DATALOADER

    def val_dataloader(self):
        return VAL_DATALOADER

if __name__ == '__main__':
    # Inicialização do modelo
    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

    # Testa um batch para verificar se está tudo funcionando
    batch = next(iter(TRAIN_DATALOADER))
    outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])

    print(outputs.logits.shape)  # Exibe a forma da saída dos logits

    # settings
    MAX_EPOCHS = 10

    logger = TensorBoardLogger("logs/", name="detr_training")

    CHECKPOINT_CALLBACK = ModelCheckpoint(
        monitor="validation_loss",  # Monitora a perda de validação
        dirpath="outputs/",     # Diretório onde o modelo será salvo
        filename="detr-{epoch:02d}-{validation_loss:.2f}",  # Padrão do nome do arquivo
        save_top_k=1,               # Apenas salva o melhor modelo
        mode="min"                  # Salva o modelo com a menor perda de validação
    )

    # Inicializa o treinador
    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=0.1,
        accumulate_grad_batches=8,
        log_every_n_steps=5,
        callbacks=[CHECKPOINT_CALLBACK],
        logger=logger,
        precision="16-mixed",
    )
    
    model.train()
    
    # Inicia o treinamento
    trainer.fit(model)
