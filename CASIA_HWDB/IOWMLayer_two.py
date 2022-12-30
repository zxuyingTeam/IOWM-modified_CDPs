# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

dtype = torch.FloatTensor

class OWMLayer:

    def __init__(self, shape_list, alpha, l2_reg_lambda, args=None):

        input_size = int(shape_list[0][0])
        hidden_size = int(shape_list[1][0])
        self.class_num = int(shape_list[1][1])
        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            self.P1 = Variable((1.0 / alpha[0]) * torch.eye(input_size).type(dtype)).to(self.device)
            self.P2 = Variable((1.0 / alpha[1]) * torch.eye(hidden_size).type(dtype)).to(self.device)
        self.w1 = get_weight(shape_list[0]).to(self.device)
        self.w2 = get_weight(shape_list[1], zeros=True).to(self.device)
        self.myAFun = nn.ReLU().to(self.device)
        self.lambda_loss = l2_reg_lambda

    def owm_learn(self, batch_x, batch_y, lr_list, immune=False):
        y1 = self.myAFun(batch_x.mm(self.w1))
        y2 = y1.mm(self.w2)

        # immune,when immune=True,don't rnew projection matrix
        r = torch.mean(y1, 0, True)
        k = torch.mm(self.P2, torch.t(r))
        if not immune:
            p2 = self.P2 - torch.mm(k, torch.t(k)) / (lr_list[1] + torch.mm(r, k))
            self.P2 = p2.detach()

        e = torch.mean(y2 - batch_y, 0, True)
        dw2 = torch.mm(k.data, e.data)
        r = torch.mean(batch_x, 0, True)
        k = torch.mm(self.P1, torch.t(r))
        if not immune:
            p1 = self.P1 - torch.mm(k, torch.t(k)) / (lr_list[2] + torch.mm(r, k))
            self.P1 = p1.detach()
        # delta ReLU
        delta = (torch.mean(y1, 0, True).data > 0).type(dtype).to(self.device)
        delta = Variable(delta, requires_grad=False)
        e = torch.mm(e, torch.t(self.w2)) * delta
        dw1 = (torch.mm(k.data, e.data))

        # Backward + Optimize
        self.w1.data -= lr_list[0] * dw1
        self.w2.data -= lr_list[0] * dw2

    def predict_labels(self, batch_x, batch_y):
        _, batch_y = torch.max(batch_y, 1)
        labels = batch_y.type(torch.LongTensor).to(self.device)
        # Forward
        y1 = batch_x.mm(self.w1)
        y1 = self.myAFun(y1)
        y_pred = y1.mm(self.w2)

        _, predicted = torch.max(y_pred.data, 1)
        correct = (predicted.cpu() == labels.cpu()).sum()
        # return correct.float() / labels.size(0)
        return correct

seed_num = 30
np.random.seed(seed_num)
torch.manual_seed(seed_num)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_num)

def get_weight(shape, zeros=None):
    if zeros is None:
        w = np.random.normal(0, 1.0, shape)
        w = torch.from_numpy(w / (np.sqrt(sum(shape) / 2.0)))
        w = w / torch.norm(w)
    else:
        w = np.zeros(shape)
        w = torch.from_numpy(w)
    return Variable(w.type(dtype), requires_grad=True)

def trans_onehot(index, batch_size_=200, class_num_=10):
    y = torch.LongTensor(batch_size_, 1).random_()
    if index.dim() == 1:
        index = index.unsqueeze(dim=1)
    y[:] = index
    y_onehot = torch.FloatTensor(batch_size_, class_num_)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot

