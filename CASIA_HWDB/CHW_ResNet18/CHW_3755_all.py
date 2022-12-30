import argparse
import os
import shutil
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warning
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # use gpu0,1
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import scipy.io as scio
import logger
from myresnet import *
from gx_folder import *

Logger = logger.Logger('./CHW_3755_all.txt')
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
dir_name = "/mydata/CASIA_HWDB_new/data/"
parser.add_argument('--data', metavar='DIR', default=dir_name + 'chinese_hand_write/',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 512)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='./model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--renow', default='./model_now.pth.tar', type=str, metavar='NOWPATH',
                    help='path to last iteration checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_false',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

parser.add_argument('--device', type=str, default='cuda:0,1')
parser.add_argument('--multi_gpu', type=bool, default=True)

from datetime import datetime

best_prec1 = 0


def main():
    global args, best_prec1

    args = parser.parse_args()
    class_num = 3755
    model = resnet18(class_num=3755)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if (args.multi_gpu) and (torch.cuda.device_count() > 1):
        Logger.append(torch.cuda.device_count())
        Logger.append('Use {} GPUs'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            Logger.append("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            Logger.append("=> loaded checkpoint '{}' (epoch {}, best_prec1 {})"
                          .format(args.resume, checkpoint['epoch'], best_prec1))
        else:
            Logger.append("=> no checkpoint found at '{}'".format(args.resume))

    # # optionally renow from a checkpoint
    # if args.renow:
    #     if os.path.isfile(args.renow):
    #         Logger.append("=> loading checkpoint '{}'".format(args.renow))
    #         checkpoint = torch.load(args.renow)
    #         args.start_epoch = checkpoint['epoch']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         Logger.append("=> loaded checkpoint '{}' (epoch {})"
    #                       .format(args.renow, checkpoint['epoch']))
    #     else:
    #         Logger.append("=> no checkpoint found at '{}'".format(args.renow))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = gx_ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(100),
            # transforms.CenterCrop(96),
            transforms.RandomResizedCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        class_num=class_num)
    # args.workers
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, )

    val_dataset = gx_ImageFolder(valdir, transforms.Compose([
        transforms.Resize(100),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        normalize,
    ]),
                                 class_num=class_num)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    Logger.append(('The train datasets and test datasets:', len(train_dataset), len(val_dataset)))
    # args.evaluate
    print(args.evaluate)
    if args.evaluate:
        save_train_data(train_loader, model, criterion)
        validate(val_loader, model, criterion, save_data=True)
        return
    # args.start_epoch
    # prev_time = datetime.now()
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        prev_time = datetime.now()
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        print(time_str)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # if i == 0:
        #     print(input.shape)
        # measure data loading time
        data_time.update(time.time() - end)
        # target = target.cuda(async=True)  
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        # output, x2 = nn.parallel.data_parallel(model, input_var)
        output, x2 = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            Logger.append('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top3=top3))


def save_train_data(train_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # target = target.cuda(async=True)  
        target = target.cuda()
        input = input.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            # compute output
            # output, x2 = nn.parallel.data_parallel(model, input_var)
            output, x2 = model(input_var)
            PATH = dir_name + 'CHW_mat3755/train'
            mat_name = 'chwdata' + str(i) + '.mat'
            os.makedirs(PATH, exist_ok=True)
            numpy_data = x2.data.cpu().numpy()
            numpy_lables = target_var.data.cpu().numpy()
            scio.savemat(os.path.join(PATH, mat_name),
                         mdict={'data': numpy_data, 'labels': numpy_lables})
            loss = criterion(output, target_var)
            # measure accuracy and record loss
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top3.update(prec3[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                Logger.append('Test: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                    i, len(train_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top3=top3))
    Logger.append(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
                  .format(top1=top1, top3=top3))

def validate(val_loader, model, criterion, save_data=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # target = target.cuda(async=True) 
        target = target.cuda()
        input = input.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            # compute output
            # output, x2 = nn.parallel.data_parallel(model, input_var)
            output, x2 = model(input_var)
            save_data = save_data
            #dir_name = "/mydata/CASIA_HWDB_new/data/"
            PATH = dir_name + 'CHW_mat3755/test'
            mat_name = 'chwdata' + str(i) + '.mat'
            if save_data:
                os.makedirs(PATH, exist_ok=True)
                numpy_data = x2.data.cpu().numpy()
                numpy_lables = target_var.data.cpu().numpy()
                scio.savemat(os.path.join(PATH, mat_name),
                             mdict={'data': numpy_data, 'labels': numpy_lables})

            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top3.update(prec3[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    Logger.append('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
        i, len(val_loader), batch_time=batch_time, loss=losses,
        top1=top1, top3=top3))

    Logger.append(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
                  .format(top1=top1, top3=top3))

    return top1.avg

def save_checkpoint(state, is_best, filename='model_now.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
