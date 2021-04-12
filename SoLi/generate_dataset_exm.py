import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from PIL import Image

class MySampler(torch.utils.data.Sampler):
    def __init__(self, end_idx, seq_length):        
        indices = []
        for i in range(len(end_idx)-1):
            start = end_idx[i]
            end = end_idx[i+1] - seq_length
            # print(start, end)
            if start >end:
                indices.append(torch.arange(start, end_idx[i+1])) # if it is smaller than the sequence length
            else:
                indices.append(torch.arange(start, end))
        indices = torch.cat(indices)
        self.indices = indices
        
    def __iter__(self):
        indices = self.indices#[torch.randperm(len(self.indices))]
        return iter(indices.tolist())
    
    def __len__(self):
        return len(self.indices)


class MyDataset(Dataset):
    def __init__(self, image_paths, seq_length, transform, length):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        
    def __getitem__(self, index):
        start = index
        end = index + self.seq_length
        # print('Getting images from {} to {}'.format(start, end))
        indices = list(range(start, end))
        images = []
        labels = []
        for i in indices:
            # if len(self.image_paths)
            image_path = self.image_paths[i][0]
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            images.append(image)
            labels.append(int(self.image_paths[i][1]))
        x = torch.stack(images)
        y = torch.tensor(labels, dtype=torch.float)
        
        return x, y
    
    def __len__(self):
        return self.length
    
def generate_dataLoader(DataFolder='train',seq_length=40,
                        batch_size=100,channel=1,root_dir='./data/'):
    class_image_paths = []
    end_idx = []
    use_channel = 'ch'+str(channel)
    file_path = root_dir+DataFolder
    for d in os.scandir(file_path):
        if d.is_dir:
            paths = glob.glob(os.path.join(d.path, use_channel+'*.jpg'))
            paths = sorted(paths, key=lambda path: int(path.split('/')[-1].split('_')[1]))
            # Add class idx to paths
            class_F = str(d.path).split('/')[-1].split('_')[0]
            # print(class_F)
            paths = [(p, class_F) for p in paths]
            class_image_paths.extend(paths)
            end_idx.extend([len(paths)])
                
    end_idx = [0, *end_idx]
    end_idx = torch.cumsum(torch.tensor(end_idx), 0)

    sampler = MySampler(end_idx, seq_length)
    transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()])

    dataset = MyDataset(
        image_paths=class_image_paths,
        seq_length=seq_length,
        transform=transform,
        length=len(sampler))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler
    )
    return loader

def generate_dataLoader(DataFolder='train',seq_length=40,
                        batch_size=100,num_channels = 4,root_dir='./data/'):
    class_image_paths = []
    end_idx = []
    channels =range(num_channels) # number oc channels we plan to use
    use_channel = 'ch'+str(channel)
    file_path = root_dir+DataFolder
    for d in os.scandir(file_path):
        if d.is_dir:
            paths = glob.glob(os.path.join(d.path, use_channel+'*.jpg'))
            paths = sorted(paths, key=lambda path: int(path.split('/')[-1].split('_')[1]))
            # Add class idx to paths
            class_F = str(d.path).split('/')[-1].split('_')[0]
            # print(class_F)
            paths = [(p, class_F) for p in paths]
            class_image_paths.extend(paths)
            end_idx.extend([len(paths)])
                
    end_idx = [0, *end_idx]
    end_idx = torch.cumsum(torch.tensor(end_idx), 0)

    sampler = MySampler(end_idx, seq_length)
    transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()])

    dataset = MyDataset(
        image_paths=class_image_paths,
        seq_length=seq_length,
        transform=transform,
        length=len(sampler))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler
    )
    return loader


num_channels = 4
root_dir='./data/'
DataFolder='train'
seq_length = 40
batch_size = 3

class_image_paths = []
end_idx = []
channels =range(num_channels) # number oc channels we plan to use
use_channel = 'ch'+str(channels[0])
file_path = root_dir+DataFolder
for d in os.scandir(file_path):
    if d.is_dir:
        paths = []
        for c in channels:
            paths += glob.glob(os.path.join(d.path, 'ch'+str(c)+'*.jpg'))
        paths = sorted(paths, key=lambda path: int(path.split('/')[-1].split('_')[1]))
        # Add class idx to paths
        class_F = str(d.path).split('/')[-1].split('_')[0]
        class_image_paths.extend(paths)
        end_idx.extend([len(paths)])
    
end_idx = [0, *end_idx]
end_idx = torch.cumsum(torch.tensor(end_idx), 0)

sampler = MySampler(end_idx, seq_length)
transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()])

dataset = MyDataset(
    image_paths=class_image_paths,
    seq_length=seq_length*num_channels,
    transform=transform,
    length=len(sampler))

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler
)

for i, (images, labels) in enumerate(loader):
    if i<1:
        print(images.shape)
        images = images.view(-1, seq_length,1024*4)
        labels = labels[:,-1].long()
    else:
        break