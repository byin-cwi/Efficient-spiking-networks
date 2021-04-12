#!/usr/bin/env python
""" Makes a data set in klepto files with MEL frequencies from the TIMIT speech corpus"""
#"""https://github.com/philipperemy/timit core test set"""#


from __future__ import print_function
import numpy as np
import os

import klepto

from preprocess_TIMIT.targets import get_target, get_timit_dict
from preprocess_TIMIT.features import get_features
from preprocess_TIMIT.speakers import get_speaker_lists


# Location of the TIMIT speech corpus
#########################################################
###             Add path to TIMIT corpus              ###
#########################################################
rootDir = "./TIMIT/"

if "TEST/" in os.listdir(rootDir):
    raise Warning("TIMIT TEST data is missing")
if "TRAIN/" in os.listdir(rootDir):
    raise Warning("TIMIT TRAIN data is missing")

# # Location of the target data set folder
# drainDir = "data_set/"


####################################
# pre process parameters

# Mel-Frequency Cepstrum Coefficients, default 12
numcep=12 #40
# the number of filters in the filterbank, default 26.
numfilt =26 #40

# the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
winlen = 0.025
# the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
winstep = 0.01
# use  first or first+second order derivation
grad = 2

para_name = "mfcc" + str(numcep) + "-" + str(numfilt) + "win" + str(int(winlen*1000)) + "-" +str(int(winstep*1000))

train_speaker, valid_speaker = get_speaker_lists(rootDir + "TRAIN/",p_valid=0.)
print(train_speaker)
print(train_speaker.__len__())
print(valid_speaker)
print(valid_speaker.__len__())
dic_location = "preprocess_TIMIT/phonemlist"

timit_dict = get_timit_dict(dic_location)

train_set_x = []
train_set_y = []
valid_set_x = []
valid_set_y = []

dirlist = ['DR5', 'DR6', 'DR7', 'DR3', 'DR2', 'DR1', 'DR4', 'DR8']
i = 0
for d in dirlist:
    for dirName, subdirList, fileList in os.walk(rootDir + "TRAIN/" + d + "/"):
        print('Found directory: %s' % dirName)

        path,folder_name = os.path.split(dirName)
        print('Speaker: ' + folder_name)
        if folder_name.__len__() >= 1:
            temp_name = ""
            for fname in sorted(fileList):
                name = fname.split(".")[0]
                if name != temp_name:
                    if 'SI' in name or 'SX' in name:
                        temp_name = name
                        print('\t%s' % dirName+"/"+name)
                        wav_location = dirName+"/"+name+".WAV"
                        phn_location = dirName+"/"+name+".PHN"
                        feat, frames, samplerate = get_features(wav_location, numcep, numfilt, winlen, winstep, grad)
                        print(feat.shape)
                        input_size = feat.shape[0]
                        target = get_target(phn_location,timit_dict, input_size)
                        if folder_name in train_speaker:
                            np.save('./npy_dataset/train/'+name+'-'+str(i)+'.npy', np.hstack((feat,target)))
                        # elif folder_name in valid_speaker:
                        #     np.save('./npy_dataset/valid/'+name+'-'+str(i)+'.npy', np.hstack((feat,target)))
                        else:
                            assert False, "unknown name"
                        i+=1
                        feat = []
                        target = []

i = 0
for d in dirlist:
    for dirName, subdirList, fileList in os.walk(rootDir + "TEST/" + d + "/"):
        print('Found directory: %s' % dirName)

        path,folder_name = os.path.split(dirName)
        print('Speaker: ' + folder_name)
        if folder_name.__len__() >= 1:
            temp_name = ""
            for fname in sorted(fileList):
                name = fname.split(".")[0]
                if name != temp_name:
                    if 'SA' not in name:
                        temp_name = name
                        print('\t%s' % dirName+"/"+name)
                        wav_location = dirName+"/"+name+".WAV"
                        phn_location = dirName+"/"+name+".PHN"
                        feat, frames, samplerate = get_features(wav_location, numcep, numfilt, winlen, winstep, grad)
                        print(feat.shape)
                        input_size = feat.shape[0]
                        target = get_target(phn_location,timit_dict, input_size)
                        np.save('./npy_dataset/test/'+name+'-'+str(i)+'.npy', np.hstack((feat,target)))
                        i+=1
                        feat = []
                        target = []


i = 0
for d in dirlist:
    for dirName, subdirList, fileList in os.walk(rootDir + "TEST_org/" + d + "/"):
        print('Found directory: %s' % dirName)

        path,folder_name = os.path.split(dirName)
        print('Speaker: ' + folder_name)
        if folder_name.__len__() >= 1:
            temp_name = ""
            for fname in sorted(fileList):
                name = fname.split(".")[0]
                if name != temp_name:
                    # if 'SA' in name:
                    temp_name = name
                    print('\t%s' % dirName+"/"+name)
                    wav_location = dirName+"/"+name+".WAV"
                    phn_location = dirName+"/"+name+".PHN"
                    feat, frames, samplerate = get_features(wav_location, numcep, numfilt, winlen, winstep, grad)
                    print(feat.shape)
                    input_size = feat.shape[0]
                    target = get_target(phn_location,timit_dict, input_size)
                    
                    np.save('./npy_dataset/valid/'+name+'-'+str(i)+'.npy', np.hstack((feat,target)))
                    i+=1
                    feat = []
                    target = []

print(np.hstack((feat,target)).shape)