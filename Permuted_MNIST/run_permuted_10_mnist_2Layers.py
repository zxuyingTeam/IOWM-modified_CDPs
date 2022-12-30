import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from datetime import datetime
from OWMLayer import OWMLayer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu

# Seed
seed_num = 30
np.random.seed(seed_num)
torch.manual_seed(seed_num)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_num)
else:
    print('[CUDA unavailable]')
    sys.exit()
# Hyper Pameters
class_num = 10  # mnist
num_epochs = 35
batch_size = 100
learning_rate = 3.0
dtype = torch.cuda.FloatTensor  # run on GPU
# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
# Data Loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

def get_weight(shape, zeros=None):
    np.random.seed(seed_num)
    if zeros is None:
        w = np.random.normal(0, 1.0, shape)
        w = torch.from_numpy(w/(np.sqrt(sum(shape)/2.0)))
    else:
        w = np.zeros(shape)
        w = torch.from_numpy(w)
    return Variable(w.type(dtype), requires_grad=True)

def get_bias(shape):
    bias = 0.01 * np.random.rand(shape)
    bias = torch.from_numpy(bias)
    return Variable(bias.type(dtype), requires_grad=True)

def get_layer(shape, alpha=0, zeros=None):
    """
    :type alpha: learningrate
    """
    w = get_weight(shape, zeros)
    return w, OWMLayer(shape, alpha)

alpha = 1.0
# Layer1
w1, force_layer1 = get_layer([28*28, 100], alpha=alpha)
b1 = get_bias(w1.size(1))
# Layer_out
w2, force_layer2 = get_layer([100, class_num], alpha=alpha)
relu = nn.ReLU().cuda()
# myDrop = nn.Dropout(p=0).cuda()
criterion = nn.CrossEntropyLoss().cuda()
n = 0
lambda_loss = 1e-3
Task_num = 10

def my_test(task_num=0):
    correct_all = []
    for task_index in range(task_num+1):
        ss = np.arange(28 * 28)
        if task_index > 0:
            np.random.seed(task_index)
            np.random.shuffle(ss)
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = Variable(images).cuda()
            images = images.view(-1, 28 * 28)
            numpy_data = images.data.cpu().numpy()
            input = torch.from_numpy(numpy_data[:, ss])
            input = Variable(input.type(dtype))
            output1 = relu(input.mm(w1) + b1)
            y_pred = output1.mm(w2)
            _, predicted = torch.max(y_pred.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()
        print('the %dth accuracy is: %0.6f' % (task_index + 1, correct.float() / total))
        correct_all.append((correct.float() / total))
    print('The average accuracy of the testset for the first %d tasks is: %0.6f ' % (task_num + 1, sum(correct_all) / len(correct_all)))
    return correct_all

prev_time = datetime.now()
for task_index in range(Task_num):
    ss = np.arange(28*28)
    if task_index > 0:
        np.random.seed(task_index)
        np.random.shuffle(ss)
    for epoch in range(num_epochs):
        correct = 0
        for i, (images, labels) in enumerate(train_loader):
            labels = Variable(labels).cuda()
            images = Variable(images).cuda()
            images = images.view(-1, 28 * 28)
            numpy_data = images.data.cpu().numpy()
            input = torch.from_numpy(numpy_data[:, ss])
            input = Variable(input.type(dtype))
            # Forward + Backward + Optimize
            output1 = relu(input.mm(w1) + b1)
            y_pred = output1.mm(w2)
            loss = criterion(y_pred, labels)+lambda_loss*(torch.norm(w1)+torch.norm(w2))
            loss.backward()
            force_layer1.force_learn(w1, input, learning_rate)
            force_layer2.force_learn(w2, output1, learning_rate)

            _, predicted = torch.max(y_pred.data, 1)
            correct += (predicted.cpu() == labels.cpu()).sum()

            if ((i + 1) % (len(train_dataset) // (batch_size))) == 0:
                print('Task [{:d}/{:d}]: Epoch [{:d}/{:d}], Iter [{:d}/{:d}] Loss: {:.3f},Accuracy: {:.6f}'
                      .format(task_index + 1, Task_num, epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size,
                              loss.item(), correct.float() / len(train_dataset)))


    my_test(task_index)
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)
    print(time_str)




