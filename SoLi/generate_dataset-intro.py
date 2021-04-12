import torchvision
import json
############ generate train-eval-test json faile ###########
data_split_file = './deep-soli-master/config/file_half.json'
with open(data_split_file) as f:
    data_split = json.load(f)

train_file = data_split['train']
eval_file = data_split['eval']

all_file = train_file+eval_file


with open('data-split.json', 'w') as f:
    json.dump(data_split, f)
    
################ move the data to each folder ###############

import shutil
import os
import json
data_split_file = './data-split.json'
with open(data_split_file) as f:
    data_split = json.load(f)

train_file = data_split['train']
eval_file = data_split['eval']
test_file = data_split['test']

file_dir = './dsp_target/'#os.getcwd() # current dir path

list_dir = os.listdir(file_dir) ## all folders
dest = './data/'#os.path.join(cur_dir,'/path/leadingto/merge_1') 

for sub_dir in list_dir:
    if sub_dir in train_file:
        dir_to_move = os.path.join(file_dir, sub_dir)
        shutil.move(dir_to_move, dest+'train/')
    elif sub_dir in test_file:
        dir_to_move = os.path.join(file_dir, sub_dir)
        shutil.move(dir_to_move, dest+'test/')
    elif sub_dir in eval_file:
        dir_to_move = os.path.join(file_dir, sub_dir)
        shutil.move(dir_to_move, dest+'eval/')

########## load dataset to pythorch
def load_dataset():
    data_path = 'data/train/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader


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
                print(start, end)
            else:
                indices.append(torch.arange(start, end))
        indices = torch.cat(indices)
        self.indices = indices
        
    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
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
        print('Getting images from {} to {}'.format(start, end))
        indices = list(range(start, end))
        images = []
        labels = []
        for i in indices:
            image_path = self.image_paths[i][0]
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            images.append(image)
            labels.append(int(self.image_paths[i][1]))
        x = torch.stack(images)
        y = torch.tensor(labels, dtype=torch.float)
        # print(self.image_paths[start][1])
        # y = torch.tensor([int(self.image_paths[start][1])], dtype=torch.long)
        
        return x, y
    
    def __len__(self):
        return self.length


root_dir = './data/'#'./data/train'
file_paths = [d.path for d in os.scandir(root_dir) if d.is_dir]

class_image_paths = []
# end_idx = [0,]
end_idx = []
use_channel = 'ch'+str(1)
for c, file_path in enumerate(file_paths[1:2]):
    print(file_path)
    for d in os.scandir(file_path):
        print(d.path)
        if d.is_dir:
            # paths = sorted(glob.glob(os.path.join(d.path, use_channel+'*.jpg')))
            paths = glob.glob(os.path.join(d.path, use_channel+'*.jpg'))
            paths = sorted(paths, key=lambda path: int(path.split('/')[-1].split('_')[1]))
            print(paths[:2])
            # Add class idx to paths
            class_F = str(d.path).split('/')[-1].split('_')[0]
            print(class_F)
            paths = [(p, class_F) for p in paths]
            class_image_paths.extend(paths)
            end_idx.extend([len(paths)])
            

##############################################################
end_idx = [0, *end_idx]
end_idx = torch.cumsum(torch.tensor(end_idx), 0)
seq_length = 40

sampler = MySampler(end_idx, seq_length)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

dataset = MyDataset(
    image_paths=class_image_paths,
    seq_length=seq_length,
    transform=transform,
    length=len(sampler))

loader = DataLoader(
    dataset,
    batch_size=100,
    sampler=sampler
)

i = 0
for data, target in loader:
    if i<10:
        print(data.shape,target.shape)
        i+=1
    else:
        break

def generate_dataLoader(DataFolder='train',seq_length=40,
                        batch_size=100,channel=1,root_dir='./data/'):
    class_image_paths = []
    end_idx = []
    use_channel = 'ch'+str(channel)
    file_path = root_dir+DataFolder
    for d in os.scandir(file_path):
        if d.is_dir:
            # paths = sorted(glob.glob(os.path.join(d.path, use_channel+'*.jpg')))
            paths = glob.glob(os.path.join(d.path, use_channel+'*.jpg'))
            paths = sorted(paths, key=lambda path: int(path.split('/')[-1].split('_')[1]))
            print(paths[:2])
            # Add class idx to paths
            class_F = str(d.path).split('/')[-1].split('_')[0]
            print(class_F)
            paths = [(p, class_F) for p in paths]
            class_image_paths.extend(paths)
            end_idx.extend([len(paths)])
                
    end_idx = [0, *end_idx]
    end_idx = torch.cumsum(torch.tensor(end_idx), 0)

    sampler = MySampler(end_idx, seq_length)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

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

train_loader = generate_dataLoader()
i = 0
for data, target in train_loader:
    target.numpy().tolist
    if i<1:
        print(data.shape,target.shape)
        i+=1
    else:
        break