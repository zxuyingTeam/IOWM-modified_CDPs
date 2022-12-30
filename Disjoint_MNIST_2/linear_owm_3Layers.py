import torch
import numpy as np

class Net(torch.nn.Module):

    def __init__(self, input_size, hidden1, hidden2, output_size, seed=79):
        super(Net, self).__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_size = output_size
        self.relu = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(self.input_size + 1, self.hidden1, bias=False)
        self.fc2 = torch.nn.Linear(self.hidden1+1, self.hidden2, bias=False)
        self.fc3 = torch.nn.Linear(self.hidden2 + 1, self.output_size, bias=False)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)


    def forward(self, x,):
        h_list = []
        h = x.view(-1, self.input_size)
        temp = torch.ones([1, 1]).cuda()
        h = torch.cat((h, temp.repeat([x.shape[0], 1])), 1)
        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc1(h))

        h = torch.cat((h, temp.repeat([h.shape[0], 1])), 1)
        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc2(h))

        h = torch.cat((h, temp.repeat([h.shape[0], 1])), 1)
        h_list.append(torch.mean(h, 0, True))
        y = self.fc3(h)
        return y, h_list
