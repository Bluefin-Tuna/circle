import numpy as np
import torch
from torch.utils.data import Dataset

class NoisyCircles(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = np.load(x)
        self.y = np.load(y)
        self.transform = transform

    def __getitem__(self, idx):
        img = self.x[idx]
        cp = self.y[idx]
        
        if self.transform:
            img = np.expand_dims(np.asarray(img), axis=0)
            img = torch.from_numpy(np.array(img, dtype=np.float32))
            cp = torch.from_numpy(np.array(np.asarray(cp), dtype=np.float32))
            img = self.transform(img)

        return img, cp

    def __len__(self):
        return len(self.x)