import os
import sys
sys.path.append("..")
import h5py
import json
import random
import timeit
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from array import *
from natsort import natsorted
# =================================================================================================

# =================================================================================================


# GET DATA FILE NAMES

# read train/test split file
factor = 0.5
version = '01'

do_maxframe = False     # Use complete frame sequence, fill with zero the remaining, max frame found to be 145
do_nrzframe = True     # Use normalize frame to till_frame = 40
do_original = True      # Use test and evaluation sets from Soli paper, note
do_cappixel = False     # Cap the number of pixel to or columns of the image

do_maxframe = True
# do_nrzframe = True
# do_original = True
do_cappixel = True

if do_original:
    load_data = './data-split.json'
else:
    dirname = 't' + str(int(factor*100)) + '-e' + str(int((1-factor)*100)) + '-' + version
    load_data = '../code/simu/' + dirname + '/file-' + dirname + '.json'

with open(load_data, 'r') as file_data:
    data = file_data.read()

# load objs
obj = json.loads(data)

# parse train and eval
train_obj = obj['train']
eval_obj = obj['eval']

# concatenate in all_obj 
all_obj = train_obj + eval_obj

# needed only for original file_half.json, shuffle anyway
all_obj = shuffle(all_obj)



# LOAD AND FLATTEN DATA
begin = timeit.default_timer()

# from npy file extracted by ../data/02-make-dataset.py
dir_path = './npy_data' 
use_channel = 3

if do_cappixel:
    til_pixel = 32*12
else:
    til_pixel = 32*32

if do_nrzframe:
    til_frame = 40
elif do_maxframe:
    max_frame = 145

targets = []
states = []
for index, smp in enumerate(all_obj):
    gst = smp.split('_')[0]
    dir_smp = natsorted(os.listdir(dir_path + '/' + gst + '/' + smp))

    if do_nrzframe:
        # read file channel[0] to find out number of frames, i.e. row
        file_ch = dir_path + '/' + gst + '/' + smp + '/' + dir_smp[0]
        sample = np.load(file_ch)
        row = sample.shape[0]
        nrow = []
        # random chose which rows will be duplicated. Use the same for all channels
        if row < til_frame:
            dif = til_frame - row
            for gap in range(dif):
                nrow.append(random.randint(0, row-1))

    # In the future insert loop for computing all channels from here
    file_ch = dir_path + '/' + gst + '/' + smp + '/' + dir_smp[use_channel]
    sample = np.load(file_ch)

    if do_cappixel:
        sample = sample[:,:til_pixel]

    # normalize to til_frame frames
    if do_nrzframe:
        row = sample.shape[0]
        if row > til_frame:
            row = til_frame
            sample = sample[:row,:]
        elif row < til_frame:
            dif = til_frame - row
            logger.info(f'{index}: {smp} \t- dif: {dif}')
            for gap in range(dif):
                dup = nrow[gap]
                # logger.info(f'dup: {dup}')
                cprow = sample[dup:dup+1,:]
                sample = np.insert(sample, dup, cprow, axis = 0)
    elif do_maxframe:
         row = sample.shape[0]
         if row < max_frame:
            dif = max_frame - row
            gap = np.zeros((dif, til_pixel))
            sample = np.concatenate((sample, gap), axis=0)

    sample = array('d', sample.ravel())
    # add to states and targets 
    states.append(sample)
    # logger.info(f'states: {np.array(states)}')
    targets.append(int(gst))

# turn to numpy needed for dimensionality reduction
states = np.array(states) #145 384
targets = np.array(targets)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
        

x_b, y_b  = unison_shuffled_copies(states,targets)
        
seq_length = 40 ### of 145
batch_size = 40


n_test = int(np.shape(targets)[0]*0.2)
n_train = np.shape(targets)[0]-n_test

x_train = x_b[0:n_train,:]
y_train = y_b[0:n_train]
x_test = x_b[n_train::,:]
y_test = y_b[n_train::]