import os
from torch.utils import data
from torchvision import datasets
from torch.utils.data import Dataset
import torch
import numpy as np

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

class my_Dataset(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):   
        x = torch.from_numpy(np.load(self.data_paths[index]))
        y_ = self.data_paths[index].split('_')[-1]
        y_ = int(y_.split('.')[0])
        y_tmp= np.array([int(y_)])
        y = torch.from_numpy(y_tmp)
        if self.transform:
            x = self.transform(x)
        return x, y

