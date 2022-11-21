import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from data_utils import SegmentationDataset
from torch.utils.data import DataLoader
from icecream import ic

from model import UNet


class LightningUNet(pl.LightningModule):
    def __init__(self, batch_size=1, lr=1e-4, num_workers=12, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        ic(self.batch_size, self.lr)
        self.num_workers = num_workers
        self.model = UNet(n_channels=3, n_classes=21)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    
    def training_step(self, batch, batch_idx):
        img, mask = batch
        img = img.to(self.device)
        mask = mask.to(self.device).long()
        
        logits = self.model(img)
        
        loss = self.loss_fn(logits, mask)
        self.log('train/loss', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.to(self.device)
        mask = mask.to(self.device).long()
        
        logits = self.model(img)
        
        loss = self.loss_fn(logits, mask)
        self.log('val/loss', loss)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def train_dataloader(self):
        ds = SegmentationDataset()
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        ds = SegmentationDataset(image_set='val')
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)