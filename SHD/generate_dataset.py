import os
import urllib.request
import gzip, shutil
# from keras.utils import get_file
import matplotlib.pyplot as plt
"""
The dataset is 48kHZ with 24bits precision
* 700 channels
* longest 1.17s
* shortest 0.316s
"""

# cache_dir=os.path.expanduser("~/data")
# cache_subdir="hdspikes"
# print("Using cache dir: %s"%cache_dir)
#
# # The remote directory with the data files
# base_url = "https://compneuro.net/datasets"

# Retrieve MD5 hashes from remote



# file_hashes = { line.split()[1]:line.split()[0] for line in lines if len(line.split())==2 }

files = ['data/shd_test.h5','data/shd_train.h5']

import tables
import numpy as np
fileh = tables.open_file(files[1], mode='r')
units = fileh.root.spikes.units
times = fileh.root.spikes.times
labels = fileh.root.labels

# This is how we access spikes and labels
index = 0
print("Times (ms):", times[index],max(times[index]))
print("Unit IDs:", units[index])
print("Label:", labels[index])


def binary_image_readout(times,units,dt = 1e-3):
    img = []
    N = int(1/dt)
    for i in range(N):
        idxs = np.argwhere(times<=i*dt).flatten()
        vals = units[idxs]
        vals = vals[vals > 0]
        vector = np.zeros(700)
        vector[700-vals] = 1
        times = np.delete(times,idxs)
        units = np.delete(units,idxs)
        img.append(vector)
    return np.array(img)

def binary_image_spatical(times,units,dt = 1e-3,dc = 10):
    img = []
    N = int(1/dt)
    C = int(700/dc)
    for i in range(N):
        idxs = np.argwhere(times<=i*dt).flatten()
        vals = units[idxs]
        vals = vals[vals > 0]
        vector = np.zeros(C)# add spacial count
        vector[700-vals] = 1
        times = np.delete(times,idxs)
        units = np.delete(units,idxs)
        img.append(vector)
    return np.array(img)


def generate_dataset(file_name,dt=1e-3):
    fileh = tables.open_file(file_name, mode='r')
    units = fileh.root.spikes.units
    times = fileh.root.spikes.times
    labels = fileh.root.labels

    # This is how we access spikes and labels
    index = 0
    print("Number of samples: ",len(times))
    X = []
    y = []
    for i in range(len(times)):
        tmp = binary_image_readout(times[i], units[i],dt=dt)
        X.append(tmp)
        y.append(labels[i])
    return np.array(X),np.array(y)

k = 125
plt.scatter(times[k],700-units[k], color="k", alpha=0.33, s=2)
plt.title("Label %i"%labels[k])
plt.xlabel('time [s]')
plt.ylabel('channel')
# plt.axis("off")
plt.show()

test_X,testy = generate_dataset(files[0],dt=4e-3)
np.save('./data/testX_4ms.npy',test_X)
np.save('./data/testY_4ms.npy',testy)

train_X,trainy = generate_dataset(files[1],dt=4e-3)
np.save('./data/trainX_4ms.npy',train_X)
np.save('./data/trainY_4ms.npy',trainy)


# # how many time steps on each sample
# l = []
# for i in range(len(times)):
#     l.append(len(set(times[i])))
# print(max(l),np.argmax(l))
# print(min(l),np.argmin(l))
# plt.hist(l,bins=20)
# plt.show()
# # the sampling frequence of spoken digits
# l = []
# for i in range(len(times)):
#     a = np.array(sorted(list(set(times[i]))))
#     n = len(a)
#     l.append(min(a[1:]-a[:n-1]))
# print(max(l),np.argmax(l))
# print(min(l),np.argmin(l))
# plt.hist(l)
# plt.show()
#
# #  how many spoken digits longer than 1s
# l = []
# ll = []
# for i in range(len(times)):
#     l.append(max(times[i]))
#     if max(times[i])>1.: ll.append(i)
# print(max(l),np.argmax(l))
# print(min(l),np.argmin(l))
# plt.hist(l,bins=20)
# plt.show()
#
# def binary_image_readout(times,units,dt = 1e-3):
#     img = []
#     N = int(1/dt)
#     for i in range(N):
#         idxs = np.argwhere(times<=i*dt).flatten()
#         vals = units[idxs]
#         vals = vals[vals>0]
#         vector = np.zeros(700)
#         vector[700-vals] = 1
#         times = np.delete(times,idxs)
#         units = np.delete(units,idxs)
#         img.append(vector)
#     return np.array(img)
# idx = 1358
# tmp = binary_image_readout(times[idx],units[idx],dt=5e-3)
# plt.imshow(tmp.T)
# plt.show()
# # A quick raster plot for one of the samples
#
#
# fig = plt.figure(figsize=(16,4))
# idx = ll[:3]#[1979,1358,626]#np.random.randint(len(times),size=3)
# for i,k in enumerate(idx):
#     ax = plt.subplot(1,3,i+1)
#     ax.scatter(times[k],700-units[k], color="k", alpha=0.33, s=2)
#     ax.set_title("Label %i"%labels[k])
#     # ax.axis("off")
# plt.show()
#
# fig = plt.figure(figsize=(16,8))
# idx = ll[16:22]#[1979,1358,626]#np.random.randint(len(times),size=3)
# for i,k in enumerate(idx):
#     ax = plt.subplot(2,3,i+1)
#     ax.scatter(times[k],700-units[k], color="k", alpha=0.33, s=2)
#     ax.set_title("Label %i"%labels[k])
#     # ax.axis("off")
#
# plt.show()
