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
from os.path import join
from opts import get_opts

opts = get_opts()

device = torch.cuda.is_available()

transform = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])
train_data = torchvision.datasets.ImageFolder('../data1',transform=transform)
max_iters = 28
batch_size = 100
learning_rate = 0.04
hidden_size = 64
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=False)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        
        self.fc1 = nn.Linear(307520, 512)
        self.fc2 = nn.Linear(512, 36)
        self.fc3 = nn.Linear(36,8) 

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x) 
        # x = torch.nn.functional.max_pool2d(x, 2, 2)
        # print('After first max pool')
        x = torch.nn.functional.relu(x)
        # x = torch.nn.functional.max_pool2d(x, 2, 2)
        # print(x.shape) 
        x = x.flatten(start_dim=1)

        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        output = self.fc3(x)
        return output

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNNModel()
optimizer = torch.optim.SGD(model.parameters(), learning_rate, 0.8)
model.to(device)
error = torch.nn.CrossEntropyLoss()

count = 0
loss_list = []
iteration_list = []
train_accuracy_list = []
test_accuracy_list = []
for epoch in range(max_iters):
    train_total_loss = 0
    test_total_loss = 0

    acc = 0
    model.train()
    for i, (input, lab) in enumerate(train_loader):
        # print('I',i)
        # print('Input',input.shape)
        # print('Lab',lab.shape)
        # input = input.reshape(batch_size,3,32,32)
        outputs = model(input)
        loss = error(outputs,lab)
        optimizer.zero_grad()
        pred = torch.argmax(outputs,axis=1)
        
        loss.backward()
        
        optimizer.step()
        train_total_loss += loss
        acc = (lab==pred).sum().item() / input.shape[0]
        train_accuracy_list.append(acc)
        
    # store loss and iteration
    # avg_acc = (acc/(input.shape[0]))*100
    loss_list.append(loss.data)
    iteration_list.append(count)
    # accuracy_list.append(avg_acc)
    if epoch % 2 == 0:
        print("itr: {:02d} \t train_loss: {:.2f} \t train_acc : {:.2f}".format(epoch,train_total_loss,np.mean(train_accuracy_list)))

plt.plot(np.arange(max_iters),train_accuracy_list)
plt.xlabel('Iterations')
plt.ylabel('Percent Accuracy')
plt.show()
