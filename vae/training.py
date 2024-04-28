from vae.utils.model import VariationalAutoEncoder
import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("imagenet-1k")
train_dataset = dataset["train"]
valid_dataset = dataset["validation"]


class VAEWrapper(torch.nn.Module):
    def __init__(self, vae):

        super.__init__()
        self.vae = vae

    def forward(self, inputs):

        x = inputs["x"]
        loss = self.vae.compute_loss(x)

        return loss


vae = VariationalAutoEncoder(encoder_decoder_depth=3, encoder_start_channels=64)
model = VAEWrapper(vae=vae)
optimizer = torch.optim.Adam(lr=3e-4, params=model.parameters())
lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer, factor=1.0, last_epoch=-1)
args = TrainingArguments(
    output_dir="/Users/leoperelli/Documents/AI Stuff/VAE/trainings",
    per_device_train_batch_size=16,
    save_strategy="epoch",
    save_total_limit=5,
)

# trainer = Trainer(
#     model=model,
#     optimizers=(optimizer, lr_scheduler)
#     train_dataset=
#     eval_dataset=
# )
