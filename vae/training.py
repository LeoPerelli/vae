from vae.utils.model import VariationalAutoEncoder
import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip, Resize, ColorJitter
from torchvision.transforms.functional import pil_to_tensor
from transformers import PretrainedConfig, PreTrainedModel
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger


def img_processor(batch):

    batch['x'] = [pil_to_tensor(transform(img)) for img in batch['image']]
    return batch

def collate_fn(images):
    x = torch.stack([img['x'] for img in images]).float()
    return x

class VAEWrapper(pl.LightningModule):
    def __init__(self, vae):
        super().__init__()
        self.model = vae

    def forward(self, batch):
        return batch

    def training_step(self, batch, batch_idx):
        return self.model.compute_loss(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(lr=3e-4, params=self.model.parameters())
    
dataset = load_dataset("poloclub/diffusiondb", '2m_random_50k')
batch_size = 32
transform = Compose([Resize((256,256)), RandomHorizontalFlip(), RandomRotation(20), ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25)])
train_val = dataset['train'].train_test_split(test_size=0.1, shuffle=True)
train_dataset = train_val['train']
val_dataset = train_val['test']
train_dataset.set_transform(img_processor)
val_dataset.set_transform(img_processor)
train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size)
val_dataloader = torch.utils.data.DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size)


vae = VariationalAutoEncoder(encoder_decoder_depth=3, encoder_start_channels=64)
model = VAEWrapper(vae=vae)

# default logger used by trainer (if tensorboard is installed)
logger = TensorBoardLogger(save_dir='/home/ec2-user/vae/trainings')
trainer = pl.Trainer(accelerator="gpu", default_root_dir='/home/ec2-user/vae/trainings', max_epochs=20, logger=logger)
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
