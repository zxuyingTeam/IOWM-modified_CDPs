import torch
import argparse
from torch.backends import cudnn
from torch.utils.data import DataLoader
import os
from loader_dcm import datasets
from torch.utils.data import sampler, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore warning

dir = "./Low_Dose/data/Mayo_abdomen/"
# dir_res = "./Low_Dose/RED_CNN/SGD_chest/"

parser = argparse.ArgumentParser()
parser.add_argument('--saved_path', type=str, default=dir+'npy_img/')
parser.add_argument('--test_patient', type=str, default='C280')
# parser.add_argument('--save_path', type=str, default=dir_res+'result3/')
parser.add_argument('--result_fig', type=bool, default=True)
parser.add_argument('--norm_range_min', type=float, default=-1024.0)
parser.add_argument('--norm_range_max', type=float, default=3072.0)
parser.add_argument('--trunc_min', type=float, default=-160)
parser.add_argument('--trunc_max', type=float, default=240.0)
parser.add_argument('--patch_stride', type=int, default=None)
parser.add_argument('--patch_size', type=int, default=None)
parser.add_argument('--transform', type=bool, default=False)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--print_iters', type=int, default=10)
parser.add_argument('--decay_iters', type=int, default=8200)
parser.add_argument('--save_iters', type=int, default=60)
parser.add_argument('--test_iters', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.25)
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--multi_gpu', type=bool, default=True)
args = parser.parse_args()


plt.rcParams['figure.figsize'] = (15.0, 15.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def show_images(train_data):
    sqrtn = 5
    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)
    i = 0
    for _, img in train_data:
        img = img.squeeze()
        H, W = img.shape
        #img = trunc(denormalize_(img.view(1, -1)))
        # img = deprocess_img(img.view(1, -1))
        # print(torch.max(img), torch.min(img))
        # plt.imshow(img.numpy())
        # plt.show()
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_aspect('equal')
        img = img.numpy()
        # print('img.type=', type(img))
        img = img.reshape([H, W])
        plt.imshow(img)
        i += 1
        if i > sqrtn ** 2 - 1:
            break
    plt.savefig('./Low-dose-CT.png')
    plt.savefig('./Low-dose-CT.pdf')
    plt.show()
    return


class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

def denormalize_(image, norm_range_max=3072.0, norm_range_min=-1024.0):
    image = image * (norm_range_max - norm_range_min) + norm_range_min
    return image

def trunc(mat, trunc_max=240.0, trunc_min=-160):
    mat[mat <= trunc_min] = trunc_min
    mat[mat >= trunc_max] = trunc_max
    return mat

def preprocess_img(x):
    x = transforms.ToTensor()(x)
    return (x - 0.5) / 0.5
def deprocess_img(x):
    return (x + 1.0) / 2.0
NUM_TRAIN = 50000
NUM_VAL = 5000
batch_size = 100
normalize = transforms.Normalize([0.5], [0.5])
transforms = transforms.Compose([
            transforms.ToTensor(),])
def get_loader(mode='test', load_mode=0, saved_path=None, test_patient=args.test_patient,
               patch_stride=None, patch_size=None,
               transform=None, batch_size=512):
    dataset_ = datasets(mode, load_mode, saved_path, test_patient, patch_stride, patch_size, transform)
    data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True)
    return data_loader

def main(args):
    cudnn.benchmark = True
    train_data = get_loader(mode='train', load_mode=0, saved_path=args.saved_path,
                test_patient=args.test_patient,
                transform=None, batch_size=1)
    # for _, img in train_data:
    #     img = img.squeeze()
    #     print('img.shape=', img.shape)
    #     break
    show_images(train_data)




if __name__ == "__main__":
    main(args)
