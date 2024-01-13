import torch
import torch.nn as nn

class RED_CNN_OWM(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN_OWM, self).__init__()
        self.conv_first = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_t_last = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        x_list = []
        residual_1 = x.clone()
        x_list.append(torch.mean(x, 0, True))
        out = self.relu(self.conv_first(x))
        x_list.append(torch.mean(out, 0, True))
        out = self.relu(self.conv2(out))
        x_list.append(torch.mean(out, 0, True))
        residual_2 = out.clone()
        out = self.relu(self.conv3(out))
        x_list.append(torch.mean(out, 0, True))
        out = self.relu(self.conv4(out))
        residual_3 = out.clone()
        x_list.append(torch.mean(out, 0, True))
        out = self.relu(self.conv5(out))

        # decoder
        x_list.append(torch.mean(out, 0, True))
        out = self.conv_1(out)
        out += residual_3
        x_list.append(torch.mean(out, 0, True))
        out = self.conv_2(self.relu(out))
        x_list.append(torch.mean(out, 0, True))
        out = self.conv_3(self.relu(out))
        out += residual_2
        x_list.append(torch.mean(out, 0, True))
        out = self.conv_4(self.relu(out))
        x_list.append(torch.mean(out, 0, True))
        out = self.conv_t_last(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return x_list, out
