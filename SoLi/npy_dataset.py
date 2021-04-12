import os
import glob

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

def generate_npydataset_multi_ch(DataFolder='train',num_channels=4,root_dir='./data/',target_dir='./npy_data/'):
    class_image_paths = []
    end_idx = []
    # use_channel = 'ch'+str(channel)
    file_path = root_dir+DataFolder
    for d in os.scandir(file_path):
        if d.is_dir:
            paths = []
            for c in range(num_channels):
                paths += glob.glob(os.path.join(d.path, 'ch'+str(c)+'*.jpg'))
            if len(paths) > 0:
                paths = sorted(paths, key=lambda path: int(path.split('/')[-1].split('_')[1]))
                # Add class idx to paths
                class_F = str(d.path).split('/')[-1].split('_')[0]
                npy_data = np.zeros((int(len(paths)/num_channels),32,32,num_channels))
                for i, p in enumerate(paths):
                    image = np.array(Image.open(p))
                    npy_data[int(i//num_channels),:,:,int(i%num_channels)] = image
                file_name_npy = target_dir+DataFolder+'/'+paths[0].split('/')[-2]+'.npy'
                np.save(target_dir+DataFolder+'/'+paths[0].split('/')[-2]+'.npy',npy_data)
    return 0            


# generate_npydataset_multi_ch(DataFolder='train')
# generate_npydataset_multi_ch(DataFolder='test')
# generate_npydataset_multi_ch(DataFolder='eval')

def generate_npydataset_single_ch(DataFolder='train',use_channel=3,root_dir='./data/',target_dir='./npy_data/'):
    class_image_paths = []
    end_idx = []
    channel = 'ch'+str(use_channel)
    file_path = root_dir+DataFolder
    for d in os.scandir(file_path):
        if d.is_dir:
            paths = glob.glob(os.path.join(d.path, channel+'*.jpg'))
            if len(paths) > 0:
                paths = sorted(paths, key=lambda path: int(path.split('/')[-1].split('_')[1]))
                # Add class idx to paths
                class_F = str(d.path).split('/')[-1].split('_')[0]
                npy_data = np.zeros((int(len(paths)),32,32))
                for i, p in enumerate(paths):
                    image = np.array(Image.open(p))
                    npy_data[int(i),:,:] = image
                file_name_npy = target_dir+DataFolder+'/'+paths[0].split('/')[-2]+'_'+channel+'.npy'
                np.save(target_dir+DataFolder+'/'+paths[0].split('/')[-2]+'_'+channel+'.npy',npy_data)
    return 0    

generate_npydataset_single_ch(DataFolder='train')
generate_npydataset_single_ch(DataFolder='test')
generate_npydataset_single_ch(DataFolder='eval')

def dataset_cutoff(dataset_dr,target_length=40):
    new_dataset = []
    new_dataset_y = []
    data_files = os.listdir(dataset_dr)
    for f in data_files:
        data_ = np.load(dataset_dr+f)
        l = data_.shape[0]
        label = int(f.split('.')[0].split('_')[0])
        data_x = range(l)
        if l<target_length:
            N = int(target_length//l)+1
            data_idx = np.tile(data_x,N)[:target_length]
            new_dataset.append(data_[data_idx,:])
            new_dataset_y.append(label)
        else: 
            N = int(l//target_length)+1
            for i in range(N): 
                if i==N-1: 
                    data_idx = data_x[-target_length:]
                    new_dataset.append(data_[data_idx,:])
                    new_dataset_y.append(label)
                else: 
                    data_idx = data_x[i*target_length:(i+1)*target_length]
                    new_dataset.append(data_[data_idx,:])
                    new_dataset_y.append(label)
                    
    print('dataset shape: ',np.array(new_dataset).shape,np.array(new_dataset_y).shape)
    return np.array(new_dataset),np.array(new_dataset_y)
new_test_dataset_x,new_test_dataset_y = dataset_cutoff('npy_data/test/')
np.save('./data_set/test_x1.npy', new_test_dataset_x[:2800])
np.save('./data_set/test_x2.npy', new_test_dataset_x[2800:])
np.save('./data_set/test_y1.npy', new_test_dataset_y[:2800])
np.save('./data_set/test_y2.npy', new_test_dataset_y[2800:])


new_train_dataset_x,new_train_dataset_y = dataset_cutoff('npy_data/train/')
np.save('./data_set/train_x.npy', new_train_dataset_x)
np.save('./data_set/train_y.npy', new_train_dataset_y)

new_eval_dataset_x,new_eval_dataset_y = dataset_cutoff('npy_data/eval/')
np.save('./data_set/valid_x.npy', new_eval_dataset_x)
np.save('./data_set/valid_y.npy', new_eval_dataset_y)

# def dataset_cutoff_ch(dataset_dir,target_length=40,use_channel=3):
#     new_dataset = []
#     new_dataset_y = []
#     channel = 'ch'+str(use_channel)
#     data_files = os.listdir(dataset_dir)
#     data_files = [f for f in data_files if channel in f ]
#     for f in data_files:
#         data_ = np.load(dataset_dir+f)
#         l = data_.shape[0]
#         label = int(f.split('.')[0].split('_')[0])
#         data_x = range(l)
#         if l<target_length:
#             N = int(target_length//l)+1
#             data_idx = np.tile(data_x,N)[:target_length]
#             new_dataset.append(data_[data_idx,:])
#             new_dataset_y.append(label)
#         else: 
#             N = int(l//target_length)+1
#             for i in range(N): 
#                 if i==N-1: 
#                     data_idx = data_x[-target_length:]
#                     new_dataset.append(data_[data_idx,:])
#                     new_dataset_y.append(label)
#                 else: 
#                     data_idx = data_x[i*target_length:(i+1)*target_length]
#                     new_dataset.append(data_[data_idx,:])
#                     new_dataset_y.append(label)
                    
#     print('dataset shape: ',np.array(new_dataset).shape,np.array(new_dataset_y).shape)
#     return np.array(new_dataset),np.array(new_dataset_y)
ch =3
new_test_dataset_x,new_test_dataset_y = dataset_cutoff_ch('npy_data/test/')
np.save('./data_set/test_x.npy', new_test_dataset_x)
np.save('./data_set/test_y.npy', new_test_dataset_y)


new_train_dataset_x,new_train_dataset_y = dataset_cutoff_ch('npy_data/train/')
np.save('./data_set/train_x.npy', new_train_dataset_x)
np.save('./data_set/train_y.npy', new_train_dataset_y)

new_eval_dataset_x,new_eval_dataset_y = dataset_cutoff_ch('npy_data/eval/')
np.save('./data_set/valid_x.npy', new_eval_dataset_x)
np.save('./data_set/valid_y.npy', new_eval_dataset_y)

def dataset_cutoff_ch(dataset_dir,target_length=40,use_channel=3):
    new_dataset = []
    new_dataset_y = []
    channel = 'ch'+str(use_channel)
    data_files = os.listdir(dataset_dir)
    data_files = [f for f in data_files if channel in f ]
    for f in data_files:
        data_ = np.load(dataset_dir+f)
        l = data_.shape[0]
        label = int(f.split('.')[0].split('_')[0])
        data_x = range(l)
        if l<target_length:
            N = target_length-l
            data_idx = data_x
            for i in range(N):
                randmF = np.random.randint(0,l-1)
                data_idx = np.insert(data_idx,randmF,randmF) 
            # data_idx = np.tile(data_x,N)[:target_length]
            new_dataset.append(data_[data_idx,:])
            new_dataset_y.append(label)
        else: 
            data_idx = data_x[:target_length]
            new_dataset.append(data_[data_idx,:])
            new_dataset_y.append(label)
                    
    print('dataset shape: ',np.array(new_dataset).shape,np.array(new_dataset_y).shape)
    return np.array(new_dataset),np.array(new_dataset_y)

ch =3
new_test_dataset_x,new_test_dataset_y = dataset_cutoff_ch('npy_data/test/',use_channel=ch)
np.save('./data_set_ch/test_x_ch'+str(ch)+'.npy', new_test_dataset_x)
np.save('./data_set_ch/test_y_ch'+str(ch)+'.npy', new_test_dataset_y)


new_train_dataset_x,new_train_dataset_y = dataset_cutoff_ch('npy_data/train/',use_channel=ch)
np.save('./data_set_ch/train_x_ch'+str(ch)+'.npy', new_train_dataset_x)
np.save('./data_set_ch/train_y_ch'+str(ch)+'.npy', new_train_dataset_y)

new_eval_dataset_x,new_eval_dataset_y = dataset_cutoff_ch('npy_data/eval/',use_channel=ch)
np.save('./data_set_ch/valid_x_ch'+str(ch)+'.npy', new_eval_dataset_x)
np.save('./data_set_ch/valid_y_ch'+str(ch)+'.npy', new_eval_dataset_y)

# N = 10
# data_x = np.arange(N)
# for i in range(3):
#     r = np.random.randint(0,N-1)
#     data_x = np.insert(data_x,r,r)
#     print(r,data_x)