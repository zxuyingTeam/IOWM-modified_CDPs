# import torch
# import numpy as np
# dtype = torch.cuda.FloatTensor  # run on GPU
# # dtype = torch.FloatTensor  # run on CPU
#
# class OWMLayer:
#     def __init__(self,  shape, alpha=0):
#         self.P = (1.0 / alpha) * np.eye(shape[0])
#     def force_learn(self, w, input_, learning_rate, alpha=1.0):  # input_(batch,input_size)
#         # input_ = input_.detach().numpy()
#         # input_ = input_.cpu().detach().numpy()
#         # self.r = np.reshape(np.mean(input_, 0), (1, len(np.mean(input_, 0))))
#         # self.k = np.dot(self.P, self.r.T)
#         # self.k = np.reshape(self.k, (len(self.k), 1))
#         # self.c = 1.0 / (alpha + np.dot(self.r, self.k))
#         # self.P -= self.c * np.dot(self.k, self.k.T)
#         self.r = torch.mean(input_, 0, True)  # [1,shape[0]],按列求平均值，True代表输入和输出有相同的维度，比如输入是二维的，那么输出也要是二维的
#         self.k = torch.mm(torch.from_numpy(self.P).type(dtype), torch.t(self.r))
#         self.c = 1.0 / (alpha + torch.mm(self.r, self.k))  # 1X1
#         temp = self.c * torch.mm(self.k, torch.t(self.k))
#         self.P -= temp.cpu().detach().numpy()
#         w.data -= learning_rate * torch.mm(torch.from_numpy(self.P).type(dtype), w.grad.data)
#         w.grad.data.zero_()
#     def predit_lable(self, input_, w,):
#         return torch.mm(input_, w)


# -*- coding: utf-8 -*-
import torch
import os
import numpy as np
from torch.autograd import Variable
dtype = torch.cuda.FloatTensor  # run on GPU
class OWMLayer:

    def __init__(self,  shape, alpha=0):

        self.input_size = shape[0]
        self.output_size = shape[1]
        self.alpha = alpha
        with torch.no_grad():
            self.P = Variable((1.0/self.alpha)*torch.eye(self.input_size).type(dtype))

    def force_learn(self, w, input_, learning_rate, alpha=1.0):  # input_(batch,input_size)
        self.r = torch.mean(input_, 0, True)
        self.k = torch.mm(self.P, torch.t(self.r))
        self.c = 1.0 / (alpha + torch.mm(self.r, self.k))  # 1X1
        P = self.P - self.c*torch.mm(self.k, torch.t(self.k))
        self.P = P.detach()
        w.data -= learning_rate * torch.mm(self.P.data, w.grad.data)
        w.grad.data.zero_()

    def predit_lable(self, input_, w,):
        return torch.mm(input_, w)
