import os

"""
The dataset is 48kHZ with 24bits precision
* 700 channels
* longest 1.s
* shortest 0.21s
"""

files = ['data1/ssc_test.h5','data1/ssc_valid.h5','data1/ssc_train.h5']

import tables
import numpy as np
fileh = tables.open_file(files[0], mode='r')
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


def generate_dataset(file_name,output_dir,dt=1e-3):
    fileh = tables.open_file(file_name, mode='r')
    units = fileh.root.spikes.units
    times = fileh.root.spikes.times
    labels = fileh.root.labels

    # This is how we access spikes and labels
    index = 0
    print("Number of samples: ",len(times))
    for i in range(len(times)):
        x_tmp = binary_image_readout(times[i], units[i],dt=dt)
        y_tmp = labels[i]
        output_file_name = output_dir+'ID:'+str(i)+'_'+str(y_tmp)+'.npy'
        np.save(output_file_name, x_tmp)
    print('done..')
    return 0


generate_dataset(files[0],output_dir='data/test/',dt=4e-3)

#generate_dataset(files[1],output_dir='data/valid/',dt=4e-3)

#generate_dataset(files[2],output_dir='data/train/',dt=4e-3)



