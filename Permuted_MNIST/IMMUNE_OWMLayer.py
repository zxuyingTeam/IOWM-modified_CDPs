import torch
from torch.autograd import Variable
dtype = torch.FloatTensor

class IOWMLayer:
    def __init__(self,  shape, alpha=0, args=None):
        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            self.P = Variable((1.0/alpha)*torch.eye(shape[0]).type(dtype)).to(self.device)

    def force_learn(self, w, input_, learning_rate, alpha=1.0, flag=True):  # input_(batch,input_size)
        if flag == True:
            self.r = torch.mean(input_, 0, True)
            self.k = torch.mm(self.P, torch.t(self.r))
            self.c = 1.0 / (alpha + torch.mm(self.r, self.k))  # 1X1
            P = self.P - self.c * torch.mm(self.k, torch.t(self.k))
            self.P = P.detach()
        w.data -= learning_rate * torch.mm(self.P.data, w.grad.data)
        w.grad.data.zero_()

    def predit_lable(self, input_, w,):
        return torch.mm(input_, w)
