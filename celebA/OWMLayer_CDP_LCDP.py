# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.io as scio

dtype = torch.FloatTensor   # data type

class OWMLayer_cdp:

    def __init__(self, shape_list, alpha, l2_reg_lambda, train_context, contextual_information=0, LCDP=True, args=None):
        """
        :param shape_list: neurons
        :param alpha: the regularization parameter of projection
        :param l2_reg_lambda: the weight of regularization
        :param train_context: bool,determine whether update the weight of encoder module
        :param contextual_information: 0--embedding vector; 1--orthogonal vector
        """
        contex_size = int(shape_list[1][0])
        self.class_num = int(shape_list[1][1])
        if args:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.LCDP = LCDP
        self.contextual_information = contextual_information
        self.w_in = self.get_weight(shape_list[0], requires_grad=False)
        # the weight of classifer
        self.w1 = self.get_weight(shape_list[1], zeros=True)
        with torch.no_grad():
            self.P1 = Variable((1.0 / alpha[0]) * torch.eye(int(shape_list[1][0])).type(dtype)).to(self.device)
        # the encoder vector
        if self.contextual_information == 0:
            # embedding vector
            wordvet = scio.loadmat('wordvet.mat')
            # the word_vec shape is:(40, 200)
            self.word_vec = wordvet['wordvet']
        elif self.contextual_information == 1:
            # orthogonal vector
            self.word_vec = np.zeros((40, 200))
            for i in range(40):
                self.word_vec[i, i * 5:(i + 1) * 5] = [1] * 5

        # bool type,determine whether update the weight of encoder module
        self.train_context = train_context
        # the weight of encoder module
        self.w_c = self.get_weight([self.word_vec.shape[1], contex_size], requires_grad=self.train_context)
        with torch.no_grad():
            self.P_c = Variable((1.0 / alpha[0]) * torch.eye(int(self.w_c.size(0))).type(dtype)).to(self.device)

        self.myAFun = nn.ReLU().to(self.device)
        self.criterion = nn.MSELoss().to(self.device)
        self.l2_reg_lambda = l2_reg_lambda

    def owm_learn(self, batch_x, batch_y, train=False, alpha_list=None, task_index=None):

        batch_x = torch.from_numpy(batch_x)
        batch_y = torch.from_numpy(batch_y)
        # the contextual information(contextual information)
        context_input = torch.from_numpy(self.word_vec[task_index, :])
        context_input = torch.unsqueeze(context_input, 0)  # add a dimension

        with torch.no_grad():
            labels = Variable(batch_y.type(dtype)).to(self.device)
            batch_x = Variable(batch_x.type(dtype)).to(self.device)
            context_input = Variable(context_input.type(dtype)).to(self.device)
        # compute the 2-norm of raws,the shape of norm_old is batch_x.shape[0]
        norm_old = torch.norm(batch_x, 2, 1)
        # the first layer doesn't update, so we don't add activation function
        if self.LCDP:
            y0 = batch_x.mm(self.w_in)
        else:
            y0 = self.myAFun(batch_x.mm(self.w_in))
        context = self.myAFun(context_input.mm(self.w_c))
        # rotator
        batch_x = y0 * context
        # Ensure the 2-norm of each feature remains unchanged
        g = (norm_old / torch.norm(batch_x, 2, 1)).repeat(batch_x.size(1), 1).type(dtype).to(self.device)
        batch_x.data *= g.data.t()
        # the output of classifier
        y_pred = batch_x.mm(self.w1)
        labels = labels.unsqueeze(1)

        if train:
            loss = self.criterion(y_pred, labels) + self.l2_reg_lambda * (torch.norm(context, p=1))
            loss.backward()
            # determine whether update the weight of encoder module
            if self.train_context:
                self.P_c = self.learning(context_input, self.P_c, self.w_c, alpha_list[0]*10, alpha_list[1])
            self.P1 = self.learning(batch_x, self.P1, self.w1, alpha_list[0], alpha_list[2])
        else:
            predicted = torch.round(y_pred.data)
            predicted = torch.squeeze(predicted, 1)
            correct = torch.eq(predicted.cpu(), batch_y).sum()
            return correct, batch_y.size(0)

    def learning(self, input_, Pro, weight, lr=0, alpha=1.0):
        r = torch.mean(input_, 0, True)
        k = torch.mm(Pro, torch.t(r))
        tmp = Pro - torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k))
        Pro = tmp.detach()
        weight.data -= lr * torch.mm(Pro.data, weight.grad.data)
        weight.grad.data.zero_()
        return Pro

    def get_weight(self, shape, zeros=None, seed=0, requires_grad=True):
        if seed is not None:
            np.random.seed(seed)
        if zeros is None:
            # Xavier
            w = np.random.normal(0, np.sqrt(2.0 / sum(shape)), shape)
        else:
            w = np.zeros(shape)
        w = torch.from_numpy(w).type(dtype).to(self.device)
        return Variable(w, requires_grad=requires_grad)

