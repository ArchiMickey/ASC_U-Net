import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, ToPILImage, Compose, Resize
from torchvision.datasets import VOCSegmentation
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, image_set = 'train') -> None:
        super().__init__()
        self.transform = Compose([
            Resize((160, 240)),
            ToTensor(),
        ])
        
        if not os.path.exists('dataset/VOCtrainval_11-May-2012.tar'):
            os.system("pip install gdown")
            os.system("gdown 1p5LRy7I1wuS6XrJ0mO47HvY-rVPt92_5")
            if not os.path.exists('dataset'):
                os.mkdir('dataset')
            os.system("mv VOCtrainval_11-May-2012.tar dataset/")
        
        if os.path.exists('dataset/VOCdevkit'):
          download = False
        else:
          download = True
        self.data = VOCSegmentation(root='dataset', year='2012', image_set=image_set, download=download, transform=self.transform, target_transform=Resize((160, 240)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
       img, mask = self.data[index]
       mask = np.array(mask)
       return img, mask
