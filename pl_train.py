import pytorch_lightning as pl
from pytorch_lightning import Trainer
from unet import LightningUNet

pl.seed_everything(42)
trainer = Trainer(devices=1, accelerator='gpu', max_epochs=100, check_val_every_n_epoch=10)
cfg = {
    'batch_size': 16,
    'lr': 1e-4,
}
model = LightningUNet(**cfg)
trainer.fit(model=model)