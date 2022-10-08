import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import math
import torch.nn.functional as F
from torch.utils import data



torch.manual_seed(2)

train_X = np.load('data/trainX_4ms.npy')
train_y = np.load('data/trainY_4ms.npy').astype(np.float)

test_X = np.load('data/testX_4ms.npy')
test_y = np.load('data/testY_4ms.npy').astype(np.float)

print('dataset shape: ', train_X.shape)
print('dataset shape: ', test_X.shape)

batch_size = 64

tensor_trainX = torch.Tensor(train_X)  # transform to torch tensor
tensor_trainY = torch.Tensor(train_y)
train_dataset = data.TensorDataset(tensor_trainX, tensor_trainY)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
tensor_testX = torch.Tensor(test_X)  # transform to torch tensor
tensor_testY = torch.Tensor(test_y)
test_dataset = data.TensorDataset(tensor_testX, tensor_testY)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

'''
STEP 2: MAKING DATASET ITERABLE
'''

decay = 0.1  # neuron decay rate
thresh = 0.5  # neuronal threshold
lens = 0.5  # hyper-parameters of approximate function
num_epochs = 20  # 150  # n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

'''
STEP 3a: CREATE spike MODEL CLASS
'''

b_j0 = 0.01  # neural threshold baseline
R_m = 1  # membrane resistance
dt = 1  #
gamma = .5  # gradient scale

# define approximate firing function

gradient_type = 'MG'
print('gradient_type: ',gradient_type)
scale = 6.
hight = 0.15
print('hight: ',hight,';scale: ',scale)

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma


# define approximate firing function

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
  
        if gradient_type == 'G':
            temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        elif gradient_type == 'MG':
            temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
                - gaussian(input, mu=lens, sigma=scale * lens) * hight \
                - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        elif gradient_type =='linear':
            temp = F.relu(1-input.abs())
        elif gradient_type == 'slayer':
            temp = torch.exp(-5*input.abs())
        return grad_input * temp.float() * gamma

act_fun_adp = ActFun_adp.apply
# tau_m = torch.FloatTensor([tau_m])

def mem_update_adp(inputs, mem, spike, tau_adp, b, tau_m, dt=1, isAdapt=1):
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    ro = torch.exp(-1. * dt / tau_adp).cuda()
    if isAdapt:
        beta = 1.8
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike
    B = b_j0 + beta * b

    mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
    inputs_ = mem - B
    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    return mem, spike, B, b


def output_Neuron(inputs, mem, tau_m, dt=1):
    """
    The read out neuron is leaky integrator without spike
    """
    # alpha = torch.exp(-1. * dt / torch.FloatTensor([30.])).cuda()
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    mem = mem * alpha + (1. - alpha) * R_m * inputs
    return mem


class RNN_custom(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_custom, self).__init__()

        self.hidden_size = hidden_size
        # self.hidden_size = input_size
        self.i_2_h1 = nn.Linear(input_size, hidden_size[0])
        self.h1_2_h1 = nn.Linear(hidden_size[0], hidden_size[0])
        self.h1_2_h2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.h2_2_h2 = nn.Linear(hidden_size[1], hidden_size[1])

        self.h2o = nn.Linear(hidden_size[1], output_size)

        self.tau_adp_h1 = nn.Parameter(torch.Tensor(hidden_size[0]))
        self.tau_adp_h2 = nn.Parameter(torch.Tensor(hidden_size[1]))
        self.tau_adp_o = nn.Parameter(torch.Tensor(output_size))
        self.tau_m_h1 = nn.Parameter(torch.Tensor(hidden_size[0]))
        self.tau_m_h2 = nn.Parameter(torch.Tensor(hidden_size[1]))
        self.tau_m_o = nn.Parameter(torch.Tensor(output_size))

        # nn.init.orthogonal_(self.h1_2_h1.weight)
        # nn.init.orthogonal_(self.h2_2_h2.weight)
        nn.init.orthogonal_(self.h1_2_h1.weight)
        nn.init.orthogonal_(self.h2_2_h2.weight)
        nn.init.xavier_uniform_(self.i_2_h1.weight)
        nn.init.xavier_uniform_(self.h1_2_h2.weight)
        nn.init.xavier_uniform_(self.h2o.weight)

        nn.init.constant_(self.i_2_h1.bias, 0)
        nn.init.constant_(self.h1_2_h2.bias, 0)
        nn.init.constant_(self.h2_2_h2.bias, 0)
        nn.init.constant_(self.h1_2_h1.bias, 0)

        nn.init.normal_(self.tau_adp_h1,150,10)
        nn.init.normal_(self.tau_adp_h2, 150,10)
        nn.init.normal_(self.tau_adp_o, 150,10)
        nn.init.normal_(self.tau_m_h1, 20.,5)
        nn.init.normal_(self.tau_m_h2, 20.,5)
        nn.init.normal_(self.tau_m_o, 20.,5)

        self.dp = nn.Dropout(0.1)

        self.b_h1 = self.b_h2 = self.b_o = 0

    def forward(self, input):
        batch_size, seq_num, input_dim = input.shape
        self.b_h1 = self.b_h2 = self.b_o = b_j0
        # mem_layer1 = spike_layer1 = torch.zeros(batch_size, self.hidden_size[0]).cuda()
        # mem_layer2 = spike_layer2 = torch.zeros(batch_size, self.hidden_size[1]).cuda()
        mem_layer1 = torch.rand(batch_size, self.hidden_size[0]).cuda()
        mem_layer2 = torch.rand(batch_size, self.hidden_size[1]).cuda()

        spike_layer1 = torch.zeros(batch_size, self.hidden_size[0]).cuda()
        spike_layer2 = torch.zeros(batch_size, self.hidden_size[1]).cuda()
        mem_output = torch.rand(batch_size, output_dim).cuda()
        output = torch.zeros(batch_size, output_dim).cuda()

        hidden_spike_ = []
        hidden_mem_ = []
        h2o_mem_ = []

        for i in range(seq_num):
            input_x = input[:, i, :]

            h_input = self.i_2_h1(input_x.float()) + self.h1_2_h1(spike_layer1)
            mem_layer1, spike_layer1, theta_h1, self.b_h1 = mem_update_adp(h_input, mem_layer1, spike_layer1,
                                                                         self.tau_adp_h1, self.b_h1,self.tau_m_h1)
            # spike_layer1 = self.dp(spike_layer1)
            h2_input = self.h1_2_h2(spike_layer1) + self.h2_2_h2(spike_layer2)
            mem_layer2, spike_layer2, theta_h2, self.b_h2 = mem_update_adp(h2_input, mem_layer2, spike_layer2,
                                                                         self.tau_adp_h2, self.b_h2, self.tau_m_h2)
            mem_output = output_Neuron(self.h2o(spike_layer2), mem_output, self.tau_m_o)
            if i > 10:
                output= output + F.softmax(mem_output, dim=1)#F.softmax(mem_output, dim=1)#

            hidden_spike_.append(spike_layer1.data.cpu().numpy())
            hidden_mem_.append(mem_layer1.data.cpu().numpy())
            h2o_mem_.append(mem_output.data.cpu().numpy())

        return output, hidden_spike_, hidden_mem_, h2o_mem_


'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = 700
hidden_dim = [128,128]  # 128
output_dim = 20
seq_dim = 250  # Number of steps to unroll
num_encode = 700
total_steps = seq_dim

model = RNN_custom(input_dim, hidden_dim, output_dim)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model.to(device)
criterion = nn.CrossEntropyLoss()
learning_rate =  1e-2

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,eps=1e-5)

# base_params = [model.i_2_h1.weight, model.i_2_h1.bias,
#                model.h1_2_h1.weight, model.h1_2_h1.bias,
#                model.h1_2_h2.weight, model.h1_2_h2.bias,
#                model.h2_2_h2.weight, model.h2_2_h2.bias,
#                model.h2o.weight, model.h2o.bias]
# optimizer = torch.optim.Adam([
#     {'params': base_params},
#     {'params': model.tau_adp_h1, 'lr': learning_rate * 5},
#     {'params': model.tau_adp_h2, 'lr': learning_rate * 5},
#     {'params': model.tau_m_h1, 'lr': learning_rate * 1},
#     {'params': model.tau_m_h2, 'lr': learning_rate * 1},
#     {'params': model.tau_m_o, 'lr': learning_rate * 1}],
#     lr=learning_rate,eps=1e-5)

scheduler = StepLR(optimizer, step_size=10, gamma=.5)


def train(model, num_epochs=150):
    acc = []
    best_accuracy = 80
    
    for epoch in range(1,num_epochs):
        loss_sum = 0
        total = 0
        correct = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, seq_dim, input_dim).requires_grad_().to(device)
            labels = labels.long().to(device)
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/logits
            outputs, _,_,_ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
            loss_sum+= loss
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()

            total += labels.size(0)
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.long().cpu()).sum()
            else:
                correct += (predicted == labels).sum()

        scheduler.step()
        accuracy = 100. * correct.numpy() / total
        # accuracy,_ = test(model, train_loader)
        ts_acc,fr = test(model,is_test=0)
        if ts_acc > best_accuracy and accuracy > 80:
            torch.save(model, './model/model_' + str(ts_acc) + '-readout-2layer-v2-4ms.pth')
            best_accuracy = ts_acc
 
        print('epoch: ', epoch, '. Loss: ', loss.item(), '. Tr Accuracy: ', accuracy, '. Ts Accuracy: ',
         ts_acc, 'Fr: ',fr)

        acc.append(accuracy)
        # if epoch %5==0:
        #     print('epoch: ', epoch, '. Loss: ', loss_sum.item()/i, 
        #             '. Tr Accuracy: ', accuracy, '. Ts Accuracy: ', ts_acc,', Fr: ',fr)
    return acc


def test(model, dataloader=test_loader,is_test=0):
    correct = 0
    total = 0
    # Iterate through test dataset
    for images, labels in dataloader:
        images = images.view(-1, seq_dim, input_dim).to(device)

        outputs, fr_,_,_ = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if torch.cuda.is_available():
            correct += (predicted.cpu() == labels.long().cpu()).sum()
        else:
            correct += (predicted == labels).sum()

    accuracy = 100. * correct.numpy() / total
    if is_test:
        print('Mean FR: ', np.array(fr_).mean())
    return accuracy, np.array(fr_).mean()


###############################
acc = train(model, num_epochs)
test_acc,fr = test(model,is_test=1)
print(' Accuracy: ', test_acc)



# dataset shape:  (8156, 250, 700)
# dataset shape:  (2264, 250, 700)
# gradient_type:  MG
# hight:  0.15 ;scale:  6.0
# device: cuda:0
# epoch:  1 . Loss:  1.4047224521636963 . Tr Accuracy:  30.946542422756252 . Ts Accuracy:  40.90106007067138 Fr:  0.10361849
# epoch:  2 . Loss:  0.5009759664535522 . Tr Accuracy:  58.26385483079941 . Ts Accuracy:  75.79505300353357 Fr:  0.14015365
# epoch:  3 . Loss:  0.8235954642295837 . Tr Accuracy:  80.44384502206964 . Ts Accuracy:  86.17491166077738 Fr:  0.12855339
# epoch:  4 . Loss:  0.1801595240831375 . Tr Accuracy:  85.93673369298676 . Ts Accuracy:  78.00353356890459 Fr:  0.12772396
# epoch:  5 . Loss:  0.6335716843605042 . Tr Accuracy:  87.84943599803826 . Ts Accuracy:  83.25971731448763 Fr:  0.122699216
# epoch:  6 . Loss:  0.4184652864933014 . Tr Accuracy:  88.46248160863168 . Ts Accuracy:  85.02650176678445 Fr:  0.12641016
# epoch:  7 . Loss:  0.11178665608167648 . Tr Accuracy:  91.66257969592938 . Ts Accuracy:  83.70141342756183 Fr:  0.11583985
# epoch:  8 . Loss:  0.24507765471935272 . Tr Accuracy:  93.80823933300637 . Ts Accuracy:  80.43286219081273 Fr:  0.11689453
# epoch:  9 . Loss:  0.02580219879746437 . Tr Accuracy:  93.6856302108877 . Ts Accuracy:  78.53356890459364 Fr:  0.11530859
# epoch:  10 . Loss:  0.132295161485672 . Tr Accuracy:  95.15693967631192 . Ts Accuracy:  87.0583038869258 Fr:  0.107161455
# epoch:  11 . Loss:  0.052720583975315094 . Tr Accuracy:  97.4619911721432 . Ts Accuracy:  87.54416961130742 Fr:  0.111710936
# epoch:  12 . Loss:  0.013912809081375599 . Tr Accuracy:  97.63364394310936 . Ts Accuracy:  88.29505300353357 Fr:  0.11046224
# epoch:  13 . Loss:  0.013542967848479748 . Tr Accuracy:  98.43060323688083 . Ts Accuracy:  84.31978798586573 Fr:  0.11176432
# epoch:  14 . Loss:  0.03696903958916664 . Tr Accuracy:  98.14860225600785 . Ts Accuracy:  85.95406360424029 Fr:  0.113291666
# epoch:  15 . Loss:  0.12882192432880402 . Tr Accuracy:  98.4428641490927 . Ts Accuracy:  84.67314487632508 Fr:  0.11321094
# epoch:  16 . Loss:  0.008622714318335056 . Tr Accuracy:  98.60225600784699 . Ts Accuracy:  87.72084805653711 Fr:  0.11448698
# epoch:  17 . Loss:  0.08007299154996872 . Tr Accuracy:  98.58999509563512 . Ts Accuracy:  85.07067137809187 Fr:  0.109397136
# epoch:  18 . Loss:  0.17344336211681366 . Tr Accuracy:  97.91564492398234 . Ts Accuracy:  86.43992932862191 Fr:  0.114264324
# epoch:  19 . Loss:  0.31412410736083984 . Tr Accuracy:  97.89112309955861 . Ts Accuracy:  90.68021201413427 Fr:  0.110308595
# Mean FR:  0.10995052
#  Accuracy:  90.7243816254417
