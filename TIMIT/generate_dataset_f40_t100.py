import os
import matplotlib.pyplot as plt
import numpy as np

timit_files = os.listdir('./TIMIT/TRAIN/')
q = 0
dirlist = ['DR5', 'DR6', 'DR7', 'DR3', 'DR2', 'DR1', 'DR4', 'DR8']
for d in dirlist:
    s = os.listdir('TIMIT/TRAIN/'+d+'/')
    # print(s)
    for i in s:
        if i != '.DS_Store':
            # print(i)
            q += len(os.listdir('TIMIT/TRAIN/'+d+'/'+i+'/'))
print(q/4,len(s))

timit_files = os.listdir('./TIMIT/VALID/')
q = 0
dirlist = ['DR5', 'DR6', 'DR7', 'DR3', 'DR2', 'DR1', 'DR4', 'DR8']
for d in dirlist:
    s = os.listdir('TIMIT/VALID/'+d+'/')
    # print(s)
    for i in s:
        if i != '.DS_Store':
            # print(i)
            q += len(os.listdir('TIMIT/VALID/'+d+'/'+i+'/'))
print(q/4,len(s))


train_files = os.listdir('npy_dataset/train/')
print(len(train_files))
test_files = os.listdir('npy_dataset/test/')
print(len(test_files))
valid_files = os.listdir('npy_dataset/valid/')
print(len(valid_files))


length_list = []
for f in train_files:
    data_ = np.load('npy_dataset2/train/'+f)
    length_list.append(data_.shape[0])

print('mean: ',np.mean(length_list))
print('max: ',np.max(length_list))
print('min: ',np.min(length_list))
print('length: ',len(length_list))

plt.hist(length_list,bins=20)
print(np.sum(np.array(length_list)<500)/len(length_list))

length_list = []
for f in valid_files:
    data_ = np.load('npy_dataset2/valid/'+f)
    length_list.append(data_.shape[0])

print('mean: ',np.mean(length_list))
print('max: ',np.max(length_list))
print('min: ',np.min(length_list))

plt.hist(length_list,bins=20)
print(np.sum(np.array(length_list)<500)/len(length_list))

length_list = []
for f in test_files:
    data_ = np.load('npy_dataset2/test/'+f)
    length_list.append(data_.shape[0])

print('mean: ',np.mean(length_list))
print('max: ',np.max(length_list))
print('min: ',np.min(length_list))

plt.hist(length_list,bins=20)
print(np.sum(np.array(length_list)<500)/len(length_list))

                
# print(len(data_temp))


def dataset_cutoff(dataset_dr,target_length=20):
    new_dataset = []
    data_files = os.listdir(dataset_dr)
    for f in data_files:
        data_ = np.load(dataset_dr+f)
        l = data_.shape[0]
        data_x = range(l)
        if l<target_length:
            # N = int(target_length//l)+1
            # data_idx = #np.tile(data_x,N)[:target_length]
            
            data_idx = np.hstack((np.arange(l),np.ones(target_length-l)*(l-1)))
            data_idx = [int(p) for p in data_idx]
            new_dataset.append(data_[data_idx,:])
        else: 
            N = int(l//target_length)+1
            for i in range(N): 
                if i==N-1: 
                    data_idx = data_x[-target_length:]
                    new_dataset.append(data_[data_idx,:])
                else: 
                    data_idx = data_x[i*target_length:(i+1)*target_length]
                    new_dataset.append(data_[data_idx,:])
    print('dataset shape: ',np.array(new_dataset).shape)
    return np.array(new_dataset)

new_test_dataset = dataset_cutoff('npy_dataset/test/',100)
np.save('./data_set/test_f40_t100.npy', new_test_dataset)

new_train_dataset = dataset_cutoff('npy_dataset/train/',100)
np.save('./data_set/train_f40_t100.npy', new_train_dataset)

new_valid_dataset = dataset_cutoff('npy_dataset/valid/',100)
# rand_idx = np.random.randint(0,26716,8000)
# new_valid_dataset[rand_idx].shape
np.save('./data_set/valid_f40_t100.npy', new_valid_dataset)
