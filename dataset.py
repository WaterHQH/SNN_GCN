# dataset.py
import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
 
    def __init__(self, n=2000, num_classes=10, image_size=224):
        self.n = n
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.randn(3, self.image_size, self.image_size)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y
