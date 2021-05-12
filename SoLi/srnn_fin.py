import os
import sys
sys.path.append("..")
import time
import numpy as np
import librosa
import scipy.io.wavfile as wav
# from tqdm import tqdm_notebook
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader
import torchvision
from torch.optim.lr_scheduler import StepLR,MultiStepLR,LambdaLR,ExponentialLR
from data import SpeechCommandsDataset,Pad, MelSpectrogram, Rescale,Normalize
from optim import RAdam

dtype = torch.float
torch.manual_seed(0) 
# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data_root = "../data/speech_commands/train"
test_data_root = "../data/speech_commands/test"

training_words = os.listdir(train_data_root)
training_words = [x for x in training_words if os.path.isdir(os.path.join(train_data_root,x))]
training_words = [x for x in training_words if os.path.isdir(os.path.join(train_data_root,x)) if x[0] != "_" ]
print("{} training words:".format(len(training_words)))
print(training_words)


testing_words = os.listdir(test_data_root)
testing_words = [x for x in testing_words if os.path.isdir(os.path.join(train_data_root,x))]
testing_words = [x for x in testing_words if os.path.isdir(os.path.join(train_data_root,x)) 
                 if x[0] != "_"]
print("{} testing words:".format(len(testing_words)))
print(testing_words)

label_dct = {k:i for i,k in enumerate(testing_words + ["_silence_", "_unknown_"])}
for w in training_words:
    label = label_dct.get(w)
    if label is None:
        label_dct[w] = label_dct["_unknown_"]

print("label_dct:")
print(label_dct)

sr = 16000
size = 16000

noise_path = os.path.join(train_data_root, "_background_noise_")
noise_files = []
for f in os.listdir(noise_path):
    if f.endswith(".wav"):
        full_name = os.path.join(noise_path, f)
        noise_files.append(full_name)
print("noise files:")
print(noise_files)

# generate silence training and validation data

silence_folder = os.path.join(train_data_root, "_silence_")
if not os.path.exists(silence_folder):
    os.makedirs(silence_folder)
    # 260 validation / 2300 training
    generate_random_silence_files(2560, noise_files, size, os.path.join(silence_folder, "rd_silence"))

    # save 260 files for validation
    silence_files = [fname for fname in os.listdir(silence_folder)]
    with open(os.path.join(train_data_root, "silence_validation_list.txt"),"w") as f:
        f.writelines("_silence_/"+ fname + "\n" for fname in silence_files[:260])


n_fft = int(30e-3*sr)
hop_length = int(10e-3*sr)
n_mels = 40
fmax = 4000
fmin = 20
delta_order = 2
stack = True

melspec = MelSpectrogram(sr, n_fft, hop_length, n_mels, fmin, fmax, delta_order, stack=stack)
pad = Pad(size)
rescale = Rescale()
normalize = Normalize()

transform = torchvision.transforms.Compose([pad,melspec,rescale])


def collate_fn(data):
    
    X_batch = np.array([d[0] for d in data])
    std = X_batch.std(axis=(0,2), keepdims=True)
    X_batch = torch.tensor(X_batch/std)
    y_batch = torch.tensor([d[1] for d in data])
    
    return X_batch, y_batch 

batch_size = 32

train_dataset = SpeechCommandsDataset(train_data_root, label_dct, transform = transform, mode="train", max_nb_per_class=None)
train_sampler = torch.utils.data.WeightedRandomSampler(train_dataset.weights,len(train_dataset.weights))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, sampler=train_sampler, collate_fn=collate_fn)

valid_dataset = SpeechCommandsDataset(train_data_root, label_dct, transform = transform, mode="valid", max_nb_per_class=None)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)

test_dataset = SpeechCommandsDataset(test_data_root, label_dct, transform = transform, mode="test")
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)

#####################################################################################################################3
# create network

from SRNN_layers.spike_dense import *
from SRNN_layers.spike_neuron import *
from SRNN_layers.spike_rnn import *
thr_func = ActFun_adp.apply  
is_bias=True
class RNN_spike(nn.Module):
    def __init__(self,):
        super(RNN_spike, self).__init__()
        n = 256
        # is_bias=False
        self.dense_1 = spike_dense(40*3,n,
                                    tauAdp_inital_std=50,tauAdp_inital=200,
                                    tauM = 20,tauM_inital_std=5,device=device,bias=is_bias)
        self.rnn_1 = spike_rnn(n,n,tauAdp_inital_std=50,tauAdp_inital=200,
                                    tauM = 20,tauM_inital_std=5,device=device,bias=is_bias)
        self.dense_2 = readout_integrator(n,12,tauM=10,tauM_inital_std=1,device=device,bias=is_bias)
        # self.dense_2 = spike_rnn(n,12,tauM=10,tauM_inital_std=1,device=device,bias=is_bias)#10

        self.thr = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.thr,5e-2)

        torch.nn.init.kaiming_normal_(self.rnn_1.recurrent.weight)
        
        torch.nn.init.xavier_normal_(self.dense_1.dense.weight)
        torch.nn.init.xavier_normal_(self.dense_2.dense.weight)
      
        if is_bias:
            torch.nn.init.constant_(self.rnn_1.recurrent.bias,0)
            torch.nn.init.constant_(self.dense_1.dense.bias,0)
            torch.nn.init.constant_(self.dense_2.dense.bias,0)

    def forward(self,input):
        b,channel,seq_length,input_dim = input.shape
        self.dense_1.set_neuron_state(b)
        self.dense_2.set_neuron_state(b)
        self.rnn_1.set_neuron_state(b)

        fr_1 = []
        fr_2 = []
        fr_3 = []
        output = 0
 
        # input_s = input
        input_s = thr_func(input-self.thr)*1.-thr_func(-self.thr-input)*1.
        for i in range(seq_length):

            input_x = input_s[:,:,i,:].reshape(b,channel*input_dim)
            mem_layer1,spike_layer1 = self.dense_1.forward(input_x)
            mem_layer2,spike_layer2 = self.rnn_1.forward(spike_layer1)
            # mem_layer3,spike_layer3 = self.dense_2.forward(spike_layer2)
            mem_layer3 = self.dense_2.forward(spike_layer2)

            output += mem_layer3
            fr_1.append(spike_layer1.detach().cpu().numpy().mean())
            fr_2.append(spike_layer2.detach().cpu().numpy().mean())
            # fr_3.append(spike_layer3.detach().cpu().numpy().mean())

        output = F.log_softmax(output/seq_length,dim=1)
        return output,[np.mean(np.abs(input_s.detach().cpu().numpy())),np.mean(fr_1),np.mean(fr_2)]

model = RNN_spike()
criterion = nn.CrossEntropyLoss()#nn.NLLLoss()

# device = torch.device("cpu")#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)
model.to(device)

def test(data_loader,is_show = 0):
    test_acc = 0.
    sum_sample = 0.
    fr_ = []
    for i, (images, labels) in enumerate(data_loader):
        images = images.view(-1,3,101, 40).to(device)
        
        labels = labels.view((-1)).long().to(device)
        predictions,fr = model(images)
        fr_.append(fr)
        _, predicted = torch.max(predictions.data, 1)
        labels = labels.cpu()
        predicted = predicted.cpu().t()

        test_acc += (predicted ==labels).sum()
        sum_sample+=predicted.numel()
    mean_FR = np.mean(fr_,axis=0)
    if is_show:
        print('Mean FR: ',mean_FR)
    return test_acc.data.cpu().numpy()/sum_sample,mean_FR


def train(epochs,criterion,optimizer,scheduler=None):
    acc_list = []
    best_acc = 0

    path = '../model/'  # .pth'
    for epoch in range(epochs):
        train_acc = 0
        sum_sample = 0
        train_loss_sum = 0
        for i, (images, labels) in enumerate(train_dataloader):
            # if i ==0:
            images = images.view(-1,3,101, 40).to(device)
 
            labels = labels.view((-1)).long().to(device)
            optimizer.zero_grad()

            predictions,_ = model(images)
            _, predicted = torch.max(predictions.data, 1)
            train_loss = criterion(predictions,labels)
            
            # print(predictions,predicted)

            train_loss.backward()
            train_loss_sum += train_loss.item()
            optimizer.step()

            labels = labels.cpu()
            predicted = predicted.cpu().t()

            train_acc += (predicted ==labels).sum()
            sum_sample+=predicted.numel()

        if scheduler:
            scheduler.step()
        train_acc = train_acc.data.cpu().numpy()/sum_sample
        valid_acc,_ = test(test_dataloader,1)
        train_loss_sum+= train_loss

        acc_list.append(train_acc)
        print('lr: ',optimizer.param_groups[0]["lr"])
        if valid_acc>best_acc and train_acc>0.890:
            best_acc = valid_acc
            torch.save(model, path+str(best_acc)[:7]+'-srnn-v3.pth')
        print(model.thr)
        print('epoch: {:3d}, Train Loss: {:.4f}, Train Acc: {:.4f},Valid Acc: {:.4f}'.format(epoch,
                                                                           train_loss_sum/len(train_dataloader),
                                                                           train_acc,valid_acc), flush=True)
    return acc_list


learning_rate = 3e-3#1.2e-2


test_acc = test(test_dataloader)
print(test_acc)
if is_bias:
    base_params = [
                    model.dense_1.dense.weight,
                    model.dense_1.dense.bias,

                    model.rnn_1.dense.weight,
                    model.rnn_1.dense.bias,
                    model.rnn_1.recurrent.weight,
                    model.rnn_1.recurrent.bias,

                    # model.dense_2.recurrent.weight,
                    # model.dense_2.recurrent.bias,
                    model.dense_2.dense.weight,
                    model.dense_2.dense.bias,
                    ]
else:
    base_params = [
                    model.dense_1.dense.weight,
                    model.rnn_1.dense.weight,
                    model.rnn_1.recurrent.weight,
                    model.dense_2.dense.weight,
                    ]

# optimizer = torch.optim.Adamax([
#                               {'params': base_params},
#                               {'params': model.dense_1.tau_m, 'lr': learning_rate * 2},  
#                               {'params': model.dense_2.tau_m, 'lr': learning_rate * 2},  
#                               {'params': model.rnn_1.tau_m, 'lr': learning_rate * 2},  
#                               {'params': model.dense_1.tau_adp, 'lr': learning_rate * 2},  
#                             #   {'params': model.dense_2.tau_adp, 'lr': learning_rate * 10},  
#                               {'params': model.rnn_1.tau_adp, 'lr': learning_rate * 2},  
#                               ],
#                         lr=learning_rate,eps=1e-5)
optimizer = torch.optim.Adam([
                              {'params': base_params, 'lr': learning_rate},
                              {'params': model.thr, 'lr': learning_rate*0.01},
                              {'params': model.dense_1.tau_m, 'lr': learning_rate * 2},  
                              {'params': model.dense_2.tau_m, 'lr': learning_rate * 2},  
                              {'params': model.rnn_1.tau_m, 'lr': learning_rate * 2},  
                              {'params': model.dense_1.tau_adp, 'lr': learning_rate * 2.},  
                            #   {'params': model.dense_2.tau_adp, 'lr': learning_rate * 10},  
                              {'params': model.rnn_1.tau_adp, 'lr': learning_rate * 2.},  
                              ],
                        lr=learning_rate)

# scheduler = StepLR(optimizer, step_size=20, gamma=.5) # 20
scheduler = StepLR(optimizer, step_size=10, gamma=.1) # 20
# epoch=0
epochs =30
# scheduler = LambdaLR(optimizer,lr_lambda=lambda epoch: 1-epoch/70)
# scheduler = ExponentialLR(optimizer, gamma=0.85)
acc_list = train(epochs,criterion,optimizer,scheduler)

test_acc = test(test_dataloader)
print(test_acc)