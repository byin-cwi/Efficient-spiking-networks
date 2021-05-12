# bi-directional srnn within pkg

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR,MultiStepLR
import math
import torch.nn.functional as F
from torch.utils import data

from SRNN_layers.spike_dense import *#spike_dense,readout_integrator
from SRNN_layers.spike_neuron import *#output_Neuron
from SRNN_layers.spike_rnn import *# spike_rnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ',device)

def normalize(data_set,Vmax,Vmin):
    return (data_set-Vmin)/(Vmax-Vmin)#+1e-6)

train_data = np.load('./f40/train_f40_t100.npy')
test_data = np.load('./f40/test_f40_t100.npy')
valid_data = np.load('./f40/valid_f40_t100.npy')


num_channels = 39
use_channels = 39
Vmax = np.max(train_data[:,:,:use_channels],axis=(0,1))
Vmin = np.min(train_data[:,:,:use_channels],axis=(0,1))
print(train_data.shape,Vmax.shape,b_j0_value)

train_x = normalize(train_data[:,:,:use_channels],Vmax,Vmin)
train_y = train_data[:,:,num_channels:]

test_x = normalize(test_data[:,:,:num_channels],Vmax,Vmin)
test_y = test_data[:,:,num_channels:]

valid_x = normalize(valid_data[:,:,:num_channels],Vmax,Vmin)
valid_y = valid_data[:,:,num_channels:]

print('input dataset shap: ',train_x.shape)
print('output dataset shap: ',train_y.shape)
_,seq_length,input_dim = train_x.shape
_,_,output_dim = train_y.shape

batch_size =16*8
# spike_neuron.b_j0_value = 1.59

torch.manual_seed(0)
def get_DataLoader(train_x,train_y,batch_size=200):
    train_dataset = data.TensorDataset(torch.Tensor(train_x), torch.Tensor(np.argmax(train_y,axis=-1)))
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

train_loader = get_DataLoader(train_x,train_y,batch_size=batch_size)
test_loader = get_DataLoader(test_x,test_y,batch_size=batch_size)
valid_loader = get_DataLoader(valid_x,valid_y,batch_size=batch_size)

class RNN_s(nn.Module):
    def __init__(self,criterion,device,delay=0):
        super(RNN_s, self).__init__()
        self.criterion = criterion
        self.delay = delay
        
        #self.network = [input_dim,128,128,256,output_dim]
        self.network = [39,256,256,output_dim]


        self.rnn_fw1 = spike_rnn(self.network[0],self.network[1],
                               tau_initializer='multi_normal',
                               tauM=[20,20,20,20],tauM_inital_std=[1,5,5,5],
                               tauAdp_inital=[200,200,250,200],tauAdp_inital_std=[5,50,100,50],
                               device=device)
        
        self.rnn_bw1 = spike_rnn(self.network[0],self.network[2],
                                tau_initializer='multi_normal',
                                tauM=[20,20,20,20],tauM_inital_std=[5,5,5,5],
                                tauAdp_inital=[200,200,150,200],tauAdp_inital_std=[5,50,30,10],
                                device=device)
        

        self.dense_mean = readout_integrator(self.network[2]+self.network[1],self.network[3],
                                    tauM=3,tauM_inital_std=1,device=device)
        

    def forward(self, input,labels=None):
        b,s,c = input.shape
        self.rnn_fw1.set_neuron_state(b)
        self.rnn_bw1.set_neuron_state(b)
        self.dense_mean.set_neuron_state(b)
        
        loss = 0
        predictions = []
        fw_spikes = []
        bw_spikes = []
        mean_tensor = 0

        for l in range(s*5):
            input_fw=input[:,l//5,:].float()
            input_bw=input[:,-l//5,:].float()

            mem_layer1, spike_layer1 = self.rnn_fw1.forward(input_fw)
            mem_layer2, spike_layer2 = self.rnn_bw1.forward(input_bw)
            fw_spikes.append(spike_layer1)
            bw_spikes.insert(0,spike_layer2)
        
        for k in range(s*5):
            bw_idx = int(k//5)*5 + (4 - int(k%5))
            second_tensor = bw_spikes[k]#[bw_idx]
            merge_spikes = torch.cat((fw_spikes[k], second_tensor), -1)
            mean_tensor += merge_spikes
            if k %5 ==4:
                mem_layer3  = self.dense_mean(mean_tensor/5.)# mean or accumulate
            
                output = F.log_softmax(mem_layer3,dim=-1)#
                predictions.append(output.data.cpu().numpy())
                if labels is not None:
                    loss += self.criterion(output, labels[:, k//5])
                mean_tensor = 0
    
        predictions = torch.tensor(predictions)
        return predictions, loss


def test(data_loader,after_num_frames=0):
    test_acc = 0.
    sum_samples = 0
    fr = []
    for i, (images, labels) in enumerate(data_loader):
        images = images.view(-1, seq_length, input_dim).to(device)
        labels = labels.view((-1,seq_length)).long().to(device)
        predictions, _ = model(images)
        _, predicted = torch.max(predictions.data, 2)
        labels = labels.cpu()
        predicted = predicted.cpu().t()
        # fr.append(fr_)
        
        test_acc += (predicted == labels).sum()
        
        sum_samples = sum_samples + predicted.numel()
    # print(predicted[1],'\n',labels[1])
    # if is_fr:
    #     print('Mean fr: ', np.mean(fr))
    return test_acc.data.cpu().numpy() / sum_samples

def test_vote(data_loader,after_num_frames=0):
    test_acc = 0.
    sum_samples = 0
    for i, (images, labels) in enumerate(data_loader):
        images = images.view(-1, seq_length, input_dim).to(device)
        labels = labels.view((-1,seq_length)).long().to(device)
        predictions, _ = model(images)
        _, predicted = torch.max(predictions.data, 2)
        labels = labels.cpu()#.data.numpy()
        sum_samples = sum_samples + predicted.numel()
        
        predicted = predicted.cpu().t().data.numpy()
        for j in range(seq_length):
            res_tsp = predicted[:,j*5:(j+1)*5]
            lab_tsp = labels[:,j]
            for k in range(len(labels)):
                if i==0 and k==1:
                    print(lab_tsp[k], res_tsp[k,:])
                counts = np.bincount(res_tsp[k,:])
                pred = np.argmax(counts)
                if pred == lab_tsp[k]:
                    test_acc += 1
                # if lab_tsp[k] in res_tsp[k,:]:
                #     test_acc += 1.

        #for j in range(5):
            #test_acc += (predicted[:,np.arange(seq_length)*5+j] == labels).sum()
        
        
    return test_acc / sum_samples*5

def train(model,loader,optimizer,scheduler=None,num_epochs=10):
    best_acc = 0
    path = 'model/'  # .pth'
    acc_list=[]
    print(model.rnn_fw1.b_j0)
    for epoch in range(num_epochs):
        train_acc = 0
        train_loss_sum = 0
        sum_samples = 0
        for i, (images, labels) in enumerate(loader):
            images = images.view(-1, seq_length, input_dim).requires_grad_().to(device)
            labels = labels.view((-1,seq_length)).long().to(device)
            optimizer.zero_grad()
    
            predictions, train_loss = model(images, labels)
            _, predicted = torch.max(predictions.data, 2)
            
            train_loss.backward()
            train_loss_sum += train_loss
            optimizer.step()

            labels = labels.cpu()
            predicted = predicted.cpu().t()
            
            train_acc += (predicted == labels).sum()
            sum_samples = sum_samples + predicted.numel()
            torch.cuda.empty_cache()
        if scheduler is not None:
            scheduler.step()
            
        train_acc = train_acc.data.cpu().numpy() / sum_samples
        valid_acc = test(valid_loader)
        
        if valid_acc>best_acc and train_acc>0.30:
            best_acc = valid_acc
            torch.save(model, path+str(best_acc)[:7]+'-bi-srnn-v3_MN-v1.pth')

        acc_list.append(train_acc)
        print('epoch: {:3d}, Train Loss: {:.4f}, Train Acc: {:.4f},Valid Acc: {:.4f}'.format(epoch,
                                                                           train_loss_sum.item()/len(loader)/(seq_length),
                                                                           train_acc,valid_acc), flush=True)
    return acc_list


num_epochs = 200
criterion = nn.NLLLoss()#nn.CrossEntropyLoss()
model = RNN_s(criterion=criterion,device=device)
model = torch.load('./model/0.65942-bi-srnn-v3_MN.pth') # v1: only MN initialize fw rnn
# model = torch.load('./model/0.64553-bi-srnn-v3_MN.pth') # v2: MN initialize fw and bw rnn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)
model.to(device)


# print(model.parameters())
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)

learning_rate =1e-3

base_params = [
               model.rnn_fw1.dense.weight,model.rnn_fw1.dense.bias,
               model.rnn_fw1.recurrent.weight,model.rnn_fw1.recurrent.bias,
               model.rnn_bw1.dense.weight,model.rnn_bw1.dense.bias,
               model.rnn_bw1.recurrent.weight,model.rnn_bw1.recurrent.bias,

               model.dense_mean.dense.weight,model.dense_mean.dense.bias]

optimizer = torch.optim.Adagrad([
    {'params': base_params},
    {'params': model.rnn_fw1.tau_adp, 'lr': learning_rate * 5},
    {'params': model.rnn_bw1.tau_adp, 'lr': learning_rate * 5},
    {'params': model.rnn_fw1.tau_m, 'lr': learning_rate * 2},
    {'params': model.rnn_bw1.tau_m, 'lr': learning_rate * 2},
    {'params': model.dense_mean.tau_m, 'lr': learning_rate * 2}],
    lr=learning_rate,eps=1e-5)
    
optimizer = torch.optim.Adamax([
    {'params': base_params}],
    lr=learning_rate)

scheduler = StepLR(optimizer, step_size=100, gamma=.5) # LIF

# training network

# with sechdual
test_acc = test(test_loader)
print(test_acc)

train_acc_list = train(model,train_loader,optimizer,scheduler,num_epochs=num_epochs)
test_acc = test(test_loader)
print(test_acc)
# print(test_vote(test_loader))


# q = 'abcdefghijklmnopqrstuvwxyz'
# fw = []
# bw = []
# for i in range(len(q)):
#     for j in range(5):
#         fw.append(q[i]+str(j))
#         bw.insert(0,q[-i-1]+str(j))

# d_bw = []        
# for k in range(len(fw)):
#     bw_idx = int(k//5)*5 + 4 - int(k%5)
#     d_bw.append(bw[bw_idx])

# print(fw)
# print(bw)
# print(d_bw)
