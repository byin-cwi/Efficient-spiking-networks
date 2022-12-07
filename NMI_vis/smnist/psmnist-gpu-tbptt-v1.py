import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR,MultiStepLR,LambdaLR
import math
import keras
from torch.utils import data
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task", help="choose the task: smnist and psmnist", type=str,default="smnist")
parser.add_argument("--ec_f", help="choose the encode function: rbf, rbf-lc, poisson", type=str,default='rbf')
parser.add_argument("--dc_f", help="choose the decode function: adp-mem, adp-spike, integrator", type=str,default='adp-spike')#'integrator')
parser.add_argument("--batch_size", help="set the batch_size", type=int,default=300)
parser.add_argument("--encoder", help="set the number of encoder", type=int,default=80)
parser.add_argument("--num_epochs", help="set the number of epoch", type=int,default=200)
parser.add_argument("--learning_rate", help="set the learning rate", type=float,default=8e-3)
parser.add_argument("--len", help="set the length of the gaussian", type=float,default=0.3)
parser.add_argument('--network', nargs='+', type=int,default=[64,256,256])

torch.manual_seed(0)
def load_dataset(task='smnist'):
    if task == 'smnist':
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    elif task == 'psmnist':
        X_train = np.load('./ps_data/ps_X_train.npy')
        X_test = np.load('./ps_data/ps_X_test.npy')
        y_train = np.load('./ps_data/Y_train.npy')
        y_test = np.load('./ps_data/Y_test.npy')
    else:
        print('only two task, -- smnist and psmnist')
        return 0
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    train_dataset = data.TensorDataset(X_train,y_train) # create train datset
    test_dataset = data.TensorDataset(X_test,y_test) # create test datset

    return train_dataset,test_dataset

'''
STEP 3a_v2: CREATE Adaptative spike MODEL CLASS
'''
b_j0 = 0.1  # neural threshold baseline
tau_m = 20  # ms membrane potential constant
R_m = 1  # membrane resistance
dt = 1  #
gamma = .5  # gradient scale
lens = .3 # .5

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma


class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
        scale = 6.0
        hight = .15
        #temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
               - gaussian(input, mu=lens, sigma=scale * lens) * hight \
               - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        # temp =  gaussian(input, mu=0., sigma=lens)
        # temp = F.relu(1-input.abs()) 93.53
        # temp = torch.exp(-5*input.abs())
        return grad_input * temp.float() * gamma


act_fun_adp = ActFun_adp.apply



def mem_update_adp(inputs, mem, spike, tau_adp,tau_m, b, dt=1, isAdapt=1):
    #     tau_adp = torch.FloatTensor([tau_adp])
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    ro = torch.exp(-1. * dt / tau_adp).cuda()
    # tau_adp is tau_adaptative which is learnable # add requiregredients
    if isAdapt:
        beta = 1.8
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike
    B = b_j0 + beta * b

    mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
    inputs_ = mem - B
    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    # spike = F.relu(inputs_)
    return mem, spike, B, b

def output_Neuron(inputs, mem, tau_m, dt=1):
    """
    The read out neuron is leaky integrator without spike
    """
    # alpha = torch.exp(-1. * dt / torch.FloatTensor([30.])).cuda()
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    mem = mem * alpha + (1. - alpha) * R_m * inputs
    return mem

'''
STEP 3b: CREATE MODEL CLASS
'''


class RNN_custom(nn.Module):
    def __init__(self, input_size, stride, hidden_dims, output_size,DC_f='mem'):
        super(RNN_custom, self).__init__()

        self.DC_f = DC_f
        self.b_j0 = b_j0
        self.stride = stride
        self.input_size = input_size
        self.output_size = output_size
        self.criterion = nn.CrossEntropyLoss()
    
        self.r1_dim = hidden_dims[0]
        self.r2_dim = hidden_dims[1]
        self.d1_dim = hidden_dims[2]
        self.i2h = nn.Linear(input_size, self.r1_dim)
        self.h2h = nn.Linear(self.r1_dim, self.r1_dim)
        self.h2d = nn.Linear(self.r1_dim, self.r2_dim)
        self.d2d = nn.Linear(self.r2_dim, self.r2_dim)
        self.dense1 = nn.Linear(self.r2_dim, self.d1_dim)
        self.d2o = nn.Linear(self.d1_dim, self.output_size)

        self.tau_adp_r1 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_adp_r2 = nn.Parameter(torch.Tensor(self.r2_dim))
        self.tau_adp_d1 = nn.Parameter(torch.Tensor(self.d1_dim))
        self.tau_adp_o = nn.Parameter(torch.Tensor(self.output_size))

        self.tau_m_r1 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_m_r2 = nn.Parameter(torch.Tensor(self.r2_dim))
        self.tau_m_d1 = nn.Parameter(torch.Tensor(self.d1_dim))
        self.tau_m_o = nn.Parameter(torch.Tensor(self.output_size))
 
        nn.init.orthogonal_(self.h2h.weight)
        nn.init.xavier_uniform_(self.i2h.weight)
        nn.init.xavier_uniform_(self.h2d.weight)
        nn.init.orthogonal_(self.d2d.weight)
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.d2o.weight)
        
        nn.init.constant_(self.i2h.bias, 0)
        nn.init.constant_(self.h2h.bias, 0)
        nn.init.constant_(self.h2d.bias, 0)
        nn.init.constant_(self.d2d.bias, 0)
        nn.init.constant_(self.dense1.bias, 0)
        nn.init.constant_(self.d2o.bias, 0)

        # nn.init.normal_(self.tau_adp_r1, 700,25)
        # nn.init.normal_(self.tau_adp_r2, 700,25)
        # nn.init.normal_(self.tau_adp_o, 700,25)
        # nn.init.normal_(self.tau_adp_d1, 700,25)
        
        nn.init.normal_(self.tau_adp_r1, 200,25)
        nn.init.normal_(self.tau_adp_r2, 200,25)
        nn.init.normal_(self.tau_adp_o, 200,25)
        nn.init.normal_(self.tau_adp_d1, 200,25)

        nn.init.normal_(self.tau_m_r1, 20,5)
        nn.init.normal_(self.tau_m_r2, 20,5)
        nn.init.normal_(self.tau_m_o, 20,1)
        nn.init.normal_(self.tau_m_d1, 20,5)

        self.b_r1 =self.b_r2 = self.b_o  = self.b_d1  = 0
    
    def compute_input_steps(self,seq_num):
        return int(seq_num/self.stride)

    def forward(self, input,labels,tbptt_steps=50,Training=True,optimizer=None):
        batch_size, seq_num, input_dim = input.shape
        self.b_r1 =self.b_r2 = self.b_o  = self.b_d1  = torch.tensor(b_j0)
        
        self.initial_neuron_states(batch_size)
        
        loss = 0
        l1_spikes = []
        l2_spikes = []
        l3_spikes = []
        out_spikes = []
        input = input/255.
        input_steps  = self.compute_input_steps(seq_num)

        for i in range(input_steps):
            start_idx = i*self.stride
            if start_idx < (seq_num - self.input_size):
                input_x = input[:, start_idx:start_idx+self.input_size, :].reshape(-1,self.input_size)
            else:
                input_x = input[:, -self.input_size:, :].reshape(-1,self.input_size)
            #print(input_x.shape)
            h_input = self.i2h(input_x.float()) + self.h2h(self.r1_spike)
            self.r1_mem, self.r1_spike, theta_r1, self.b_r1 = mem_update_adp(h_input,self.r1_mem, self.r1_spike, self.tau_adp_r1, self.tau_m_r1,self.b_r1)

            d_input = self.h2d(self.r1_spike) + self.d2d(self.r2_spike)
            self.r2_mem, self.r2_spike, theta_r2, self.b_r2 = mem_update_adp(d_input, self.r2_mem, self.r2_spike, self.tau_adp_r2,self.tau_m_r2, self.b_r2)

            self.d1_mem, self.d1_spike, theta_d1, self.b_d1 = mem_update_adp(self.dense1(self.r2_spike), self.d1_mem, self.d1_spike, self.tau_adp_d1,self.tau_m_d1, self.b_d1)            

            o_input = self.d2o(self.d1_spike)
            if self.DC_f[:3]=='adp':
                self.d2o_mem, self.d2o_spike, theta_o, self.b_o = mem_update_adp(o_input,self.d2o_mem, self.d2o_spike, self.tau_adp_o, self.tau_m_o, self.b_o)
                
            elif self.DC_f == 'integrator':
                self.d2o_mem = output_Neuron(o_input,self.d2o_mem, self.tau_m_o)
                
            
            
            if i > 0: 
                if self.DC_f == 'adp-mem':
                    self.output_sumspike = self.output_sumspike + F.softmax(self.d2o_mem,dim=1)
                elif self.DC_f =='adp-spike':
                    self.output_sumspike = self.output_sumspike + self.d2o_spike
                elif self.DC_f =='integrator':
                    self.output_sumspike =self.output_sumspike+ F.softmax(self.d2o_mem,dim=1)

                if Training and i % tbptt_steps==0:
                    loss = self.criterion(self.output_sumspike,labels)
                    # print(i,loss)
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    save_spike = (self.r1_spike.detach(),self.r2_spike.detach(),self.d1_spike.detach(),self.d2o_spike.detach(),self.output_sumspike.detach())
                    save_threshold = (self.b_r1.detach(),self.b_r2.detach(),self.b_d1.detach(),self.b_o.detach())
                    save_mem = (self.r1_mem.detach(),self.r2_mem.detach(),self.d1_mem.detach(),self.d2o_mem.detach())

                    self.reset_neuron_states(save_mem,save_spike,save_threshold)

                    optimizer.zero_grad()

            l1_spikes.append(self.r1_spike.detach().cpu().numpy())
            l2_spikes.append(self.r2_spike.detach().cpu().numpy())
            l3_spikes.append(self.d1_spike.detach().cpu().numpy())
            out_spikes.append(self.d2o_spike.detach().cpu().numpy())

        return self.output_sumspike,  [l1_spikes,l2_spikes,l3_spikes,out_spikes]
    
    def initial_neuron_states(self,batch_size):
        self.r1_mem = self.r1_spike = torch.rand(batch_size, self.r1_dim).cuda()*self.b_j0
        self.r2_mem = self.r2_spike = torch.rand(batch_size, self.r2_dim).cuda()*self.b_j0
        self.d1_mem = self.d1_spike = torch.rand(batch_size, self.d1_dim).cuda()*self.b_j0
        self.d2o_mem = torch.rand(batch_size, output_dim).cuda()*self.b_j0
        self.d2o_spike = self.output_sumspike = torch.zeros(batch_size, output_dim).cuda()

    def reset_neuron_states(self,saved_mem,saved_spike,saved_threshold): 
        self.r1_mem,self.r2_mem,self.d1_mem,self.d2o_mem =saved_mem
        
        self.b_r1,self.b_r2,self.b_d1,self.b_o = saved_threshold
        self.r1_spike,self.r2_spike,self.d1_spike,self.d2o_spike,self.output_sumspike = saved_spike



def train(model, num_epochs,train_loader,test_loader,file_name,MyFile):
    acc = []
    
    best_accuracy = 80
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.view(-1, seq_dim, input_dim).requires_grad_().to(device)
            labels = labels.long().to(device)
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/logits
            outputs, _ = model(images,labels,tbptt_steps=200,Training=True,optimizer=optimizer)
            # Calculate Loss: softmax --> cross entropy loss
            loss = model.criterion(outputs, labels)
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.long().cpu()).sum()
            accuracy = 100. * correct.numpy() / total
        scheduler.step()
        # accuracy = test(model, train_loader)
        ts_acc = test(model,test_loader)
        if ts_acc > best_accuracy and accuracy > 80:
            torch.save(model, './model/model_' + str(ts_acc) + '_'+file_name+'-multi_input.pth')
            best_accuracy = ts_acc
        acc.append(accuracy)
        res_str = 'epoch: '+str(epoch)+' Loss: '+ str(loss.item())+'. Tr Accuracy: '+ str(accuracy)+ '. Ts Accuracy: '+str(ts_acc)
        print(res_str)
        MyFile.write(res_str)
        MyFile.write('\n')
    return acc


def test1(model, dataloader):
    correct = 0
    total = 0
    # Iterate through test dataset
    for images, labels in dataloader:
        images = images.view(-1, seq_dim, input_dim).to(device)

        outputs, states = model(images,labels,Training=False)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if torch.cuda.is_available():
            correct += (predicted.cpu() == labels.long().cpu()).sum()
        else:
            correct += (predicted == labels).sum()

    accuracy = 100. * correct.numpy() / total
    return accuracy

def test(model, dataloader):
    correct = 0
    total = 0
    # Iterate through test dataset
    fr1 = []
    fr2 = []
    fr3 = []
    fr4 = []
    for images, labels in dataloader:
        images = images.view(-1, seq_dim, input_dim).to(device)

        outputs, states = model(images,labels,Training=False)
        r1_spike_np = np.array(states[0])
        r2_spike_np = np.array(states[1])
        d1_spike_np = np.array(states[2])
        d2_spike_np = np.array(states[3])
        fr1.append(np.mean(d1_spike_np))
        fr2.append(np.mean(r1_spike_np))
        fr3.append(np.mean(r2_spike_np))
        fr4.append(np.mean(d2_spike_np))

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if torch.cuda.is_available():
            correct += (predicted.cpu() == labels.long().cpu()).sum()
        else:
            correct += (predicted == labels).sum()

    print('fr: ',np.mean(fr1),np.mean(fr2),np.mean(fr3),np.mean(fr4))
    accuracy = 100. * correct.numpy() / total
    return accuracy

def predict(model,test_loader):
    # Iterate through test dataset
    result = np.zeros(1)
    for images, labels in test_loader:
        images = images.view(-1, seq_dim, input_dim).to(device)

        outputs, _,_,_ = model(images)
        # _, Predicted = torch.max(outputs.data, 1)
        # result.append(Predicted.data.cpu().numpy())
        predicted_vec = outputs.data.cpu().numpy()
        Predicted = predicted_vec.argmax(axis=1)
        result = np.append(result,Predicted)
    return np.array(result[1:]).flatten()

if __name__ == '__main__':
    args = parser.parse_args()

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    task = args.task
    EC_f = args.ec_f
    DC_f = args.dc_f
    num_encode=args.encoder

    train_dataset,test_dataset = load_dataset(task)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


    input_dim = 1
    input_size=1
    stride = 1
    hidden_dims = args.network#[256,128]
    output_dim = 10
    seq_dim = int(784 / input_dim)  # Number of steps to unroll

    model = RNN_custom(input_size, stride,hidden_dims, output_dim,DC_f=DC_f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:",device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    learning_rate = args.learning_rate

    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if EC_f == 'rbf-lc':
        base_params = [model.i2h.weight, model.i2h.bias, 
                model.h2h.weight, model.h2h.bias, 
                model.h2d.weight, model.h2d.bias,
                model.d2d.weight, model.d2d.bias, 
                model.d2o.weight, model.d2o.bias,model.threshold_event]
    else:
        base_params = [model.i2h.weight, model.i2h.bias, 
                model.h2h.weight, model.h2h.bias, 
                model.h2d.weight, model.h2d.bias,
                model.d2d.weight, model.d2d.bias, 
                model.dense1.weight, model.dense1.bias, 
                model.d2o.weight, model.d2o.bias]

    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': model.tau_adp_r1, 'lr': learning_rate * 5},
        {'params': model.tau_adp_r2, 'lr': learning_rate * 5},
        {'params': model.tau_adp_d1, 'lr': learning_rate * 5},
        {'params': model.tau_adp_o, 'lr': learning_rate * 5},
        {'params': model.tau_m_r1, 'lr': learning_rate * 2},
        {'params': model.tau_m_r2, 'lr': learning_rate * 2},
        {'params': model.tau_m_d1, 'lr': learning_rate * 2},
        {'params': model.tau_m_o, 'lr': learning_rate * 2},],
        lr=learning_rate)


    # scheduler = StepLR(optimizer, step_size=25, gamma=.75)
    #scheduler = MultiStepLR(optimizer, milestones=[25,50,100,150],gamma=0.5)
    # scheduler = MultiStepLR(optimizer, milestones=[50,100,150],gamma=0.5)
    scheduler = LambdaLR(optimizer,lr_lambda=lambda epoch:1-epoch/200)

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    print('Time: ',dt_string)
    file_name = 'Task-'+task+'||Time-'+ dt_string+'||EC_f--'+EC_f+'||DC_f--'+DC_f+'||multiinput'
    MyFile=open('./result_file/'+file_name+'.txt','w')
    MyFile.write(file_name)
    MyFile.write('\nnetwork: ['+str(hidden_dims[0])+' '+str(hidden_dims[1])+']')
    MyFile.write('\nlearning_rate: '+str(learning_rate))
    MyFile.write('\nbatch_size: '+str(batch_size))
    MyFile.write('\n\n =========== Result ======== \n')

    acc = train(model, num_epochs,train_loader,test_loader,file_name,MyFile)
    accuracy = test(model,test_loader)
    print('test Accuracy: ', accuracy)
    MyFile.write('test Accuracy: '+ str(accuracy))
    MyFile.close()

    ###################
    ##  Accuracy  curve
    ###################
    if num_epochs > 10:
        plt.plot(acc)
        plt.title('Learning Curve -- Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy: %')
        plt.show()


