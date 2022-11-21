import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from icecream import ic

from data_utils import SegmentationDataset
from model import UNet


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = UNet(n_channels=3, n_classes=21)
loss_fn = nn.CrossEntropyLoss(ignore_index=255)

epoch = 100
lr = 1e-4
batch_size = 4
opt = torch.optim.Adam(model.parameters(), lr=lr)
train_ds = SegmentationDataset()
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

def train():
    epoch_losses = []
    model.to(DEVICE)
    
    model.train()
    for e in range(epoch):
        losses = []
        dl_loop = tqdm(train_dl, desc=f'Epoch {e+1}/{epoch}', leave=False)
        for batch in dl_loop:
            img, mask = batch
            img = img.to(DEVICE)
            mask = torch.from_numpy(np.array(mask)).long().to(DEVICE)
            
            logits = model(img)
            loss = loss_fn(logits, mask)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            loss = loss.cpu().detach()
            losses.append(loss.item())
            dl_loop.set_postfix({'loss': loss.item()})
            
        epoch_loss = np.mean(losses)
        epoch_losses.append(epoch_loss)
        if (e+1) % 10 == 0:
            ic(f'Epoch {e+1}/{epoch} Loss: {epoch_loss}')
    
    return {
        'model': model,
        'ep_losses': epoch_losses
    }

if __name__ == "__main__":
    train()
