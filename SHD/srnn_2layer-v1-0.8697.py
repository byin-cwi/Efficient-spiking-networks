import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import math
import torch.nn.functional as F
from torch.utils import data

# train_X = np.load('data/trainX_10ms.npy')
# train_y = np.load('data/trainY_10ms.npy').astype(np.float)

# test_X = np.load('data/testX_10ms.npy')
# test_y = np.load('data/testY_10ms.npy').astype(np.float)

train_X = np.load('data/trainX_4ms.npy')
train_y = np.load('data/trainY_4ms.npy').astype(np.float)

test_X = np.load('data/testX_4ms.npy')
test_y = np.load('data/testY_4ms.npy').astype(np.float)

print('dataset shape: ', train_X.shape)
print('dataset shape: ', test_X.shape)

batch_size = 120

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
num_epochs = 50  # 150  # n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

'''
STEP 3a: CREATE spike MODEL CLASS
'''

b_j0 = 0.01  # neural threshold baseline
R_m = 1  # membrane resistance
dt = 1  #
gamma = .5  # gradient scale


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
        scale = 6.0
        hight = .15
        # temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
               - gaussian(input, mu=lens, sigma=scale * lens) * hight \
               - gaussian(input, mu=-lens, sigma=scale * lens) * hight
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

        nn.init.orthogonal_(self.h1_2_h1.weight)
        nn.init.orthogonal_(self.h2_2_h2.weight)
        nn.init.xavier_uniform_(self.i_2_h1.weight)
        nn.init.xavier_uniform_(self.h1_2_h2.weight)
        nn.init.xavier_uniform_(self.h2_2_h2.weight)
        nn.init.xavier_uniform_(self.h2o.weight)

        nn.init.constant_(self.i_2_h1.bias, 0)
        nn.init.constant_(self.h1_2_h2.bias, 0)
        nn.init.constant_(self.h2_2_h2.bias, 0)
        nn.init.constant_(self.h1_2_h1.bias, 0)

        # nn.init.constant_(self.tau_adp_h1, 50)
        # nn.init.constant_(self.tau_adp_h2, 100)
        # nn.init.constant_(self.tau_adp_o, 100)
        # nn.init.constant_(self.tau_m_h1, 10.)
        # nn.init.constant_(self.tau_m_h2, 10.)
        # nn.init.constant_(self.tau_m_o, 15.)
        # # got 85.11 [128,128]
        # nn.init.normal_(self.tau_adp_h1, 50,5)
        # nn.init.normal_(self.tau_adp_h2, 50,5)
        # nn.init.normal_(self.tau_adp_o, 50,5)
        # nn.init.normal_(self.tau_m_h1, 20.,5)
        # nn.init.normal_(self.tau_m_h2, 20.,5)
        # nn.init.normal_(self.tau_m_o, 20.,5)
        # got 0.8697 [218,128]
        nn.init.normal_(self.tau_adp_h1, 50,5)
        nn.init.normal_(self.tau_adp_h2, 50,5)
        nn.init.normal_(self.tau_adp_o, 80,5)
        nn.init.normal_(self.tau_m_h1, 20.,5)
        nn.init.normal_(self.tau_m_h2, 20.,5)
        nn.init.normal_(self.tau_m_o, 20.,5)

        self.b_h1 = self.b_h2 = self.b_o = 0

    def forward(self, input):
        batch_size, seq_num, input_dim = input.shape
        self.b_h1 = self.b_h2 = self.b_o = b_j0

        mem_layer1 = spike_layer1 = torch.rand(batch_size, self.hidden_size[0]).cuda()
        mem_layer2 = spike_layer2 = torch.rand(batch_size, self.hidden_size[1]).cuda()
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
            h2_input = self.h1_2_h2(spike_layer1) + self.h2_2_h2(spike_layer2)
            mem_layer2, spike_layer2, theta_h2, self.b_h2 = mem_update_adp(h2_input, mem_layer2, spike_layer2,
                                                                         self.tau_adp_h2, self.b_h2, self.tau_m_h2)
            mem_output = output_Neuron(self.h2o(spike_layer2), mem_output, self.tau_m_o)
            if i > 40:
                output= output + F.softmax(mem_output, dim=1)

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
learning_rate = 1e-2  # 1e-2

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
base_params = [model.i_2_h1.weight, model.i_2_h1.bias,
               model.h1_2_h1.weight, model.h1_2_h1.bias,
               model.h1_2_h2.weight, model.h1_2_h2.bias,
               model.h2_2_h2.weight, model.h2_2_h2.bias,
               model.h2o.weight, model.h2o.bias]
optimizer = torch.optim.Adam([
    {'params': base_params},
    {'params': model.tau_adp_h1, 'lr': learning_rate * 5},
    {'params': model.tau_adp_h2, 'lr': learning_rate * 5},
    {'params': model.tau_adp_o, 'lr': learning_rate * 5},
    {'params': model.tau_m_h1, 'lr': learning_rate * 2},
    {'params': model.tau_m_h2, 'lr': learning_rate * 2},
    {'params': model.tau_m_o, 'lr': learning_rate * 2}],
    lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=.5)


def train(model, num_epochs=150):
    acc = []
    best_accuracy = 80
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, seq_dim, input_dim).requires_grad_().to(device)
            labels = labels.long().to(device)
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/logits
            outputs, _,_,_ = model(images)
            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()
        scheduler.step()
        accuracy = test(model, train_loader)
        ts_acc = test(model)
        if ts_acc > best_accuracy and accuracy > 80:
            torch.save(model, './model/model_' + str(ts_acc) + '-readout-2layer-v1-12Feb[128,128]-4ms.pth')
            best_accuracy = ts_acc
        acc.append(accuracy)
        print('epoch: ', epoch, '. Loss: ', loss.item(), '. Tr Accuracy: ', accuracy, '. Ts Accuracy: ', ts_acc)
    return acc


def test(model, dataloader=test_loader):
    correct = 0
    total = 0
    # Iterate through test dataset
    for images, labels in dataloader:
        images = images.view(-1, seq_dim, input_dim).to(device)

        outputs, _,_,_ = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if torch.cuda.is_available():
            correct += (predicted.cpu() == labels.long().cpu()).sum()
        else:
            correct += (predicted == labels).sum()

    accuracy = 100. * correct.numpy() / total
    return accuracy


def predict(model):
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


###############################
acc = train(model, num_epochs)
accuracy = test(model)
print(' Accuracy: ', accuracy)

###################
##  Accuracy  curve
###################
if num_epochs > 10:
    plt.plot(acc)
    plt.title('Learning Curve -- Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy: %')
    plt.show()

from sklearn.metrics import confusion_matrix

predicted = predict(model)
print(predicted.shape, test_y.shape)
cm = confusion_matrix(test_y, predicted)
print('max output acc:', np.sum(test_y == predicted) / len(test_y))
# print(cm)
plt.imshow(cm)
plt.show()

# classification report results
from sklearn import metrics

# print("classification report :", metrics.classification_report(test_y, predicted))
# example
images = tensor_trainX[1, :, :]
images = images.view(-1, seq_dim, input_dim).to(device)


outputs, _,_,output_mem = model(images)
output_mem = np.array(output_mem).reshape(100, 20)

for i in range(20):
    plt.plot(output_mem[:, i], label=str(i))
plt.legend()
plt.show()

# best 86.97
# epoch:  0 . Loss:  2.11820387840271 . Tr Accuracy:  29.0706228543 . Ts Accuracy:  32.7296819788
# epoch:  1 . Loss:  1.1697977781295776 . Tr Accuracy:  58.9995095635 . Ts Accuracy:  60.8215547703
# epoch:  2 . Loss:  1.22520112991333 . Tr Accuracy:  70.5492888671 . Ts Accuracy:  65.4151943463
# epoch:  3 . Loss:  0.7427058815956116 . Tr Accuracy:  74.9509563512 . Ts Accuracy:  72.2614840989
# epoch:  4 . Loss:  0.7235404253005981 . Tr Accuracy:  78.0407062285 . Ts Accuracy:  72.5706713781
# epoch:  5 . Loss:  0.5518863797187805 . Tr Accuracy:  83.8155958803 . Ts Accuracy:  78.0035335689
# epoch:  6 . Loss:  0.4285832345485687 . Tr Accuracy:  86.2555174105 . Ts Accuracy:  74.5583038869
# epoch:  7 . Loss:  0.24908654391765594 . Tr Accuracy:  88.1804806278 . Ts Accuracy:  76.0600706714
# epoch:  8 . Loss:  0.3548576235771179 . Tr Accuracy:  90.3261402648 . Ts Accuracy:  81.0070671378
# epoch:  9 . Loss:  0.5398231148719788 . Tr Accuracy:  89.4801373222 . Ts Accuracy:  78.1360424028
# epoch:  10 . Loss:  0.3219108581542969 . Tr Accuracy:  93.0725846003 . Ts Accuracy:  79.6378091873
# epoch:  11 . Loss:  0.155569389462471 . Tr Accuracy:  95.3898970083 . Ts Accuracy:  82.5530035336
# epoch:  12 . Loss:  0.08130384236574173 . Tr Accuracy:  96.6650318784 . Ts Accuracy:  83.480565371
# epoch:  13 . Loss:  0.13739757239818573 . Tr Accuracy:  94.8871996077 . Ts Accuracy:  83.6572438163
# epoch:  14 . Loss:  0.14705879986286163 . Tr Accuracy:  96.3339872487 . Ts Accuracy:  83.4363957597
# epoch:  15 . Loss:  0.04227015748620033 . Tr Accuracy:  96.6772927906 . Ts Accuracy:  86.925795053
# epoch:  16 . Loss:  0.13525082170963287 . Tr Accuracy:  95.5860716037 . Ts Accuracy:  82.332155477
# epoch:  17 . Loss:  0.12825466692447662 . Tr Accuracy:  97.6091221187 . Ts Accuracy:  83.6130742049
# epoch:  18 . Loss:  0.07115688174962997 . Tr Accuracy:  97.5232957332 . Ts Accuracy:  85.2915194346
# epoch:  19 . Loss:  0.04170788452029228 . Tr Accuracy:  97.3639038744 . Ts Accuracy:  86.4399293286
# epoch:  20 . Loss:  0.0239779781550169 . Tr Accuracy:  98.1240804316 . Ts Accuracy:  84.7614840989
# epoch:  21 . Loss:  0.037288814783096313 . Tr Accuracy:  98.7371260422 . Ts Accuracy:  84.3639575972
# epoch:  22 . Loss:  0.02986038103699684 . Tr Accuracy:  99.2030407062 . Ts Accuracy:  85.9098939929
# epoch:  23 . Loss:  0.01828932762145996 . Tr Accuracy:  99.3501716528 . Ts Accuracy:  86.9699646643
# epoch:  24 . Loss:  0.018206153064966202 . Tr Accuracy:  98.8229524277 . Ts Accuracy:  85.4240282686
# epoch:  25 . Loss:  0.036419741809368134 . Tr Accuracy:  99.3869543894 . Ts Accuracy:  86.4840989399
# epoch:  26 . Loss:  0.023401712998747826 . Tr Accuracy:  99.2520843551 . Ts Accuracy:  84.5406360424
# epoch:  27 . Loss:  0.03308366984128952 . Tr Accuracy:  98.9946051986 . Ts Accuracy:  86.9699646643
# epoch:  28 . Loss:  0.02464187517762184 . Tr Accuracy:  99.2643452673 . Ts Accuracy:  84.8498233216
# epoch:  29 . Loss:  0.04858698695898056 . Tr Accuracy:  99.362432565 . Ts Accuracy:  85.9982332155
# epoch:  30 . Loss:  0.009458443149924278 . Tr Accuracy:  99.6199117214 . Ts Accuracy:  86.1749116608
# epoch:  31 . Loss:  0.00918611977249384 . Tr Accuracy:  99.6812162825 . Ts Accuracy:  85.203180212
# epoch:  32 . Loss:  0.011284046806395054 . Tr Accuracy:  99.6812162825 . Ts Accuracy:  85.203180212
# epoch:  33 . Loss:  0.013372816145420074 . Tr Accuracy:  99.6689553703 . Ts Accuracy:  86.4840989399
# epoch:  34 . Loss:  0.008697131648659706 . Tr Accuracy:  99.7425208436 . Ts Accuracy:  86.351590106
# epoch:  35 . Loss:  0.011760781519114971 . Tr Accuracy:  99.7793035802 . Ts Accuracy:  86.3074204947
# epoch:  36 . Loss:  0.009751549921929836 . Tr Accuracy:  99.8406081412 . Ts Accuracy:  86.2632508834
# epoch:  37 . Loss:  0.014502196572721004 . Tr Accuracy:  99.7915644924 . Ts Accuracy:  84.3639575972
# epoch:  38 . Loss:  0.0174576323479414 . Tr Accuracy:  99.7793035802 . Ts Accuracy:  85.9098939929
# epoch:  39 . Loss:  0.007627849001437426 . Tr Accuracy:  99.8038254046 . Ts Accuracy:  86.1749116608
# epoch:  40 . Loss:  0.012209193781018257 . Tr Accuracy:  99.767042668 . Ts Accuracy:  85.4240282686
# epoch:  41 . Loss:  0.012491744011640549 . Tr Accuracy:  99.8773908779 . Ts Accuracy:  85.3798586572
# epoch:  42 . Loss:  0.005408081691712141 . Tr Accuracy:  99.8528690535 . Ts Accuracy:  85.777385159
# epoch:  43 . Loss:  0.005841945763677359 . Tr Accuracy:  99.9264345267 . Ts Accuracy:  85.3798586572
# epoch:  44 . Loss:  0.007102818228304386 . Tr Accuracy:  99.8773908779 . Ts Accuracy:  85.9540636042
# epoch:  45 . Loss:  0.007089360151439905 . Tr Accuracy:  99.9019127023 . Ts Accuracy:  85.3356890459
# epoch:  46 . Loss:  0.0046874405816197395 . Tr Accuracy:  99.8038254046 . Ts Accuracy:  85.203180212
# epoch:  47 . Loss:  0.005372228100895882 . Tr Accuracy:  99.8651299657 . Ts Accuracy:  86.0424028269
# epoch:  48 . Loss:  0.01163505669683218 . Tr Accuracy:  99.9386954389 . Ts Accuracy:  85.6007067138
# epoch:  49 . Loss:  0.004141906276345253 . Tr Accuracy:  99.8406081412 . Ts Accuracy:  85.5565371025
