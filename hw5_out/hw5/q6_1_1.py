import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches
import os
import scipy.io
from util import *
from nn import *

device = torch.device('cpu')

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']
train_x = torch.from_numpy(train_x).type(torch.float)
train_y = torch.from_numpy(train_y).type(torch.LongTensor)

max_iters = 150
batch_size = 32
learning_rate = 0.04
hidden_size = 64
batches = get_random_batches(train_x, train_y, batch_size)
train_data_load = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x,train_y),batch_size,True)

train_acc = []
train_loss = []
model = torch.nn.Sequential(torch.nn.Linear(train_x.shape[1],hidden_size),torch.nn.Sigmoid(),torch.nn.Linear(hidden_size,train_y.shape[1]))
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.8)
        
for itr in range(max_iters):
    total_loss = 0
    acc = 0
    for data in train_data_load:
        xb = torch.autograd.Variable(data[0])
        labels = torch.autograd.Variable(data[1])
        yb = torch.argmax(labels,axis=1)

        probs = model(xb)
        pred = torch.argmax(probs,axis=1)

        loss = torch.nn.functional.cross_entropy(probs,yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss
        acc += (yb.data==pred).sum().item()
        
    avg_acc = (acc/train_y.shape[0])*100
    train_loss.append(total_loss)  
    train_acc.append(avg_acc)
    
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))

plt.plot(np.arange(max_iters),train_acc)
plt.xlabel('Iterations')
plt.ylabel('Percent Accuracy')
plt.show()




