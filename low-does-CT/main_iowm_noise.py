import torch
import argparse
from torch.backends import cudnn
from torch.utils.data import DataLoader
import os
from loader_dcm_noise import datasets
from solver_iowm import Solver

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu

dir = "./Low_Dose/data/Mayo_abdomen/"
dir_res = "./Low_Dose/IOWM/"

parser = argparse.ArgumentParser()
parser.add_argument('--saved_path', type=str, default=dir+'npy_img_Poisson_noise_1e5/')
parser.add_argument('--test_patient', type=str, default='L056')
parser.add_argument('--save_path', type=str, default=dir_res+'Poisson_noise_1e5/nosie_2/')
parser.add_argument('--result_fig', type=bool, default=True)
parser.add_argument('--norm_range_min', type=float, default=-1024.0)
parser.add_argument('--norm_range_max', type=float, default=3072.0)
parser.add_argument('--trunc_min', type=float, default=-160)
parser.add_argument('--trunc_max', type=float, default=240.0)
parser.add_argument('--patch_stride', type=int, default=64)
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--transform', type=bool, default=False)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--print_iters', type=int, default=10)
parser.add_argument('--decay_iters', type=int, default=12180)
parser.add_argument('--save_iters', type=int, default=60)
parser.add_argument('--test_iters', type=int, default=155)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--multi_gpu', type=bool, default=True)
args = parser.parse_args()
def get_loader(mode='test', load_mode=0,shuffle=True,
               saved_path=None, test_patient=args.test_patient,
               patch_stride=None, patch_size=None,
               transform=None, batch_size=512, num_workers=args.num_workers):
    dataset_ = datasets(mode, load_mode, saved_path, test_patient, patch_stride, patch_size, transform)
    data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader

def main(args):
    cudnn.benchmark = True
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.result_fig:
        fig_path = os.path.join(args.save_path, 'fig')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))
    train_loader = get_loader(mode='train', load_mode=0, saved_path=args.saved_path,
                test_patient=args.test_patient, patch_stride=args.patch_stride, patch_size=args.patch_size,
                transform=args.transform, batch_size=512, num_workers=args.num_workers)
    test_loader = get_loader(mode='test', load_mode=0, shuffle=False, saved_path=args.saved_path,
                test_patient=args.test_patient, patch_stride=None, patch_size=None,
                transform=args.transform, batch_size=1, num_workers=args.num_workers)
    print('len(train_loader):', len(train_loader))
    print('len(test_loader):', len(test_loader))

    solver = Solver(args, out_ch=96)
    solver.train(train_loader)
    solver.test(test_loader)

if __name__ == "__main__":
    main(args)
