from vae.utils.model import VariationalAutoEncoder, SimpleVariationalAutoEncoder
from vae.training import VAEWrapper
import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip, Resize, ColorJitter
from torchvision.transforms.functional import pil_to_tensor
from transformers import PretrainedConfig, PreTrainedModel
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt



# vae = SimpleVariationalAutoEncoder()
vae = VariationalAutoEncoder(encoder_decoder_depth=2, encoder_start_channels=32, img_channels=1, img_dim=28)
model = VAEWrapper.load_from_checkpoint('/home/ec2-user/vae/trainings/lightning_logs/version_73/checkpoints/epoch=15-step=7504.ckpt', vae=vae)

model = model.eval()

dataset = load_dataset("mnist")['test'].shuffle().select(range(10))
transform = Compose([Resize((28,28))])
def img_processor_2(batch):
    batch['x'] = [pil_to_tensor(transform(img)) for img in batch['image']]
    return batch
dataset.set_transform(img_processor_2)

for i in range(10):

    x = model.model.generate(x=dataset[i]['x'].float().to('cuda:0')[None])
    print(x)
    ax = plt.imshow(x[0].detach().cpu().permute(1, 2, 0))
    plt.colorbar()
    ax.figure.savefig(f'/home/ec2-user/vae/trainings/lightning_logs/version_73/{i}.png')
    plt.close()
