import sys, argparse
import numpy as np
import torch
import utils
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu


prev_time = datetime.now()
########################################################################################################################

# Seed
seed = 79
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
else:
    print('[CUDA unavailable]')
    sys.exit()

import immune_owm_2Layer as approach
import linear_owm_2Layers as network

########################################################################
# Load
print('Load data...')
data, taskcla, inputsize = utils.get(tasks=1, paths='split_10')
print('Input size =', inputsize, '\nTask info =', taskcla)

# Inits
print('Inits...')
net = network.Net(784, 800, 10).cuda()

batch_size = 40
lr = 0.2
task_num = 10
appr = approach.Appr(net, sbatch=batch_size, lr=lr, tasks=task_num)
print('-'*100)

# Loop tasks
for t, ncla in taskcla:
    print('*'*100)
    print('Task {:2d} ({:s})'.format(t+1, data[t]['name']))
    print('*'*100)

    xtrain = data[t]['train']['x'].cuda()
    ytrain = data[t]['train']['y'].cuda()
    xvalid = data[t]['test']['x'].cuda()
    yvalid = data[t]['test']['y'].cuda()

    # Train
    if t > 2:
        nums = 1
    else:
        nums = 0
    appr.train(t, xtrain, ytrain, xvalid, yvalid, data, num_epochs=20, nums=nums)
    print('-'*100)

    # Test
    print("Test on Previous Datasets:")
    correct = 0
    total = 0
    for u in range(t+1):
        xtest = data[u]['test']['x'].cuda()
        ytest = data[u]['test']['y'].cuda()
        _, test_acc = appr.eval(xtest, ytest)
        total += xtest.size(0)
        correct += round(xtest.size(0)*test_acc)
        # print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.2f}% <<<'.format(u, data[u]['name'], test_loss, 100*test_acc))
    print("Test:->>>[{:d}/{:d}], acc: {:.2f} %".format(t + 1, task_num, 100*correct/total))
xtest = data[10]['test']['x'].cuda()
ytest = data[10]['test']['y'].cuda()
_, owm_acc = appr.eval(xtest, ytest)
print("accu_owm {:g} %\n".format(owm_acc * 100))
print('*'*100)
cur_time = datetime.now()
h, remainder = divmod((cur_time - prev_time).seconds, 3600)
m, s = divmod(remainder, 60)
time_str = "Time %02d:%02d:%02d" % (h, m, s)
print(time_str)
print('Done!')




