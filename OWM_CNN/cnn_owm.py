import torch


class Net(torch.nn.Module):

    def __init__(self, inputsize):
        super(Net, self).__init__()
        ncha, size, _ = inputsize
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2)
        self.drop1 = torch.nn.Dropout(0.2)
        self.padding = torch.nn.ReplicationPad2d(1)   # 边界填充大小是(1,1,1,1)，边界是什么就用什么进行填充

        self.c1 = torch.nn.Conv2d(ncha, 64, kernel_size=2, stride=1, padding=0, bias=False)
        self.c2 = torch.nn.Conv2d(64, 128, kernel_size=2, stride=1,  padding=0, bias=False)
        self.c3 = torch.nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0, bias=False)

        self.fc1 = torch.nn.Linear(256 * 4 * 4, 1000, bias=False)
        self.fc2 = torch.nn.Linear(1000, 1000, bias=False)
        self.fc3 = torch.nn.Linear(1000, 10,  bias=False)

        torch.nn.init.xavier_normal_(self.fc1.weight)   # Xavier初始化方式，正态分布
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)

        return

    def forward(self, x,):
        h_list = []
        x_list = []
        # Gated
        x = self.padding(x)  # 图片尺寸34*34

        x_list.append(torch.mean(x, 0, True))
        con1 = self.drop1(self.relu(self.c1(x)))  # 图片尺寸33*33
        con1_p = self.maxpool(con1)  # 图片尺寸16*16

        con1_p = self.padding(con1_p)  # 18 * 18
        x_list.append(torch.mean(con1_p, 0, True))
        con2 = self.drop1(self.relu(self.c2(con1_p)))  # 图片尺寸17*17
        con2_p = self.maxpool(con2)  # 图片尺寸8*8

        con2_p = self.padding(con2_p)  # 10*10
        x_list.append(torch.mean(con2_p, 0, True))
        con3 = self.drop1(self.relu(self.c3(con2_p)))  # 9*9
        con3_p = self.maxpool(con3)  # 4*4

        h = con3_p.view(x.size(0), -1)
        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc1(h))

        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc2(h))

        h_list.append(torch.mean(h, 0, True))
        y = self.fc3(h)
        return y, h_list, x_list
