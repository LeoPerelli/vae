from vae.utils.model import VariationalAutoEncoder
import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip, Resize, ColorJitter
from torchvision.transforms.functional import pil_to_tensor
from transformers import PretrainedConfig, PreTrainedModel
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt




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

vae = VariationalAutoEncoder(encoder_decoder_depth=3, encoder_start_channels=64)
model = VAEWrapper.load_from_checkpoint('/home/ec2-user/vae/trainings/lightning_logs/version_0/checkpoints/epoch=3-step=5628.ckpt', vae=vae)

model = model.eval()


for i in range(10):

    x = model.model.generate()
    ax = plt.imshow(x[0].detach().cpu().permute(1, 2, 0))
    ax.figure.savefig(f'/home/ec2-user/vae/trainings/lightning_logs/version_0/{i}.png')
