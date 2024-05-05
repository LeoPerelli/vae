from vae.utils.model import VariationalAutoEncoder, SimpleVariationalAutoEncoder
import torch
from datasets import load_dataset
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip, Resize, ColorJitter, ToTensor
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger


def img_processor(batch):
    batch['x'] = [transform(img) for img in batch['img']]
    return batch

def img_processor_2(batch):

    batch['x'] = [transform(img).flatten(start_dim=1) for img in batch['image']]
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
        estimated_reconstruction_likelihood, kl_divergence = self.model.compute_loss(batch)
        self.log("kl_divergence", kl_divergence.mean(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("estimated_reconstruction_likelihood", estimated_reconstruction_likelihood.mean(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        loss = (kl_divergence - estimated_reconstruction_likelihood).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def on_train_epoch_end(self):

        for module_name, module_activations in self.model.activations.items():
            if module_activations == []:
                continue
            activations = torch.cat(module_activations, dim=0).flatten()
            self.logger.experiment.add_histogram(module_name, activations, self.current_epoch)
            self.model.activations[module_name].clear()

        total_norm = 0 
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        self.log('total_norm', total_norm ** 0.5)

        batch = next(iter(self.trainer.train_dataloader))
        y = self.model.generate(x=batch.to('cuda'))

        self.logger.experiment.add_images('gt', batch[0:5], global_step=self.current_epoch, dataformats='NCHW')
        self.logger.experiment.add_images('reconstructed', y[0:5], global_step=self.current_epoch, dataformats='NCHW')


    # def validation_step(self, batch, batch_idx):
    #     estimated_reconstruction_likelihood, kl_divergence = self.model.compute_loss(batch)
    #     self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     return loss


    def configure_optimizers(self):
        return torch.optim.Adam(lr=1e-3, params=self.model.parameters())
    
# dataset = load_dataset("poloclub/diffusiondb", '2m_random_50k')
# batch_size = 32
# img_dim = 32
# transform = Compose([Resize((img_dim,img_dim)), RandomHorizontalFlip(), RandomRotation(20), ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25), ToTensor()])
# train_val = dataset['train'].train_test_split(test_size=0.1, shuffle=True)
# train_dataset = train_val['train']
# val_dataset = train_val['test']
# train_dataset.set_transform(img_processor)
# val_dataset.set_transform(img_processor)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
# val_dataloader = torch.utils.data.DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size)

dataset = load_dataset("cifar10")
batch_size = 128
img_dim = 28
train_dataset = dataset['train']
val_dataset = dataset['test']
transform = Compose([Resize((img_dim,img_dim)), ToTensor()])
train_dataset.set_transform(img_processor)
val_dataset.set_transform(img_processor)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

# vae = SimpleVariationalAutoEncoder()
vae = VariationalAutoEncoder(encoder_decoder_depth=2, encoder_start_channels=32, img_dim=28, img_channels=3)
model = VAEWrapper(vae=vae)

# default logger used by trainer (if tensorboard is installed)
logger = TensorBoardLogger(save_dir='/home/ec2-user/vae/trainings')
trainer = pl.Trainer(accelerator="gpu", default_root_dir='/home/ec2-user/vae/trainings', overfit_batches=0,max_epochs=100, logger=logger, log_every_n_steps=50, gradient_clip_val=1)
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
