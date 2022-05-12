import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import os
import scipy.io
from util import *
from nn import *
import torch 
import torch.nn as nn 
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import *
import torch.optim

device = torch.cuda.is_available()

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']
train_x = torch.tensor(train_x).float()
train_y = torch.tensor(train_y)
label = np.where(train_y == 1)[1]
label = torch.tensor(label)

# train_data1 = np.hstack((train_x,train_y))
# train_loader, test_loader = torch.utils.data.DataLoader(train_data1, batch_size=16, shuffle=True), torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True)

max_iters = 50
batch_size = 100
learning_rate = 0.04
hidden_size = 64
train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(train_x,label),batch_size=batch_size,shuffle=True)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        
        self.fc1 = nn.Linear(980, 512)
        self.fc2 = nn.Linear(512, 36) 

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x) 
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        # print('After first max pool')
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        # print(x.shape) 
        x = x.flatten(start_dim=1)
        x = torch.nn.functional.relu(self.fc1(x))
        output = self.fc2(x)
        return output

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNNModel()
optimizer = torch.optim.SGD(model.parameters(), learning_rate, 0.8)
model.to(device)
error = torch.nn.CrossEntropyLoss()

count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(max_iters):
    total_loss = 0
    acc = 0
    model.train()
    for i, (input, lab) in enumerate(train_loader):
        # print('I',i)
        # print('Input',input.shape)
        # print('Lab',lab.shape)
        input = input.reshape(batch_size,1,32,32)
        outputs = model(input)
        loss = error(outputs,lab)
        optimizer.zero_grad()
        pred = torch.argmax(outputs,axis=1)
        
        loss.backward()
        
        optimizer.step()
        total_loss += loss
        acc += (lab==pred).sum().item()
        
    # store loss and iteration
    avg_acc = (acc/train_y.shape[0])*100
    loss_list.append(loss.data)
    iteration_list.append(count)
    accuracy_list.append(avg_acc)


    if epoch % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(epoch,total_loss,avg_acc))