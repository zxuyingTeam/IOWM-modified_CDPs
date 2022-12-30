import sys
import numpy as np
import torch
import utils
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu

########################################################################################################################
prev_time = datetime.now()
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

########################################################################################################################

# Load
print('Load data...')
data, taskcla, inputsize = utils.get()
print('Input size =', inputsize, '\nTask info =', taskcla)

# Inits
net = network.Net(784, 800, 10).cuda()

batch_size = 40
lr = 0.20  # leraing rate
appr = approach.Appr(net, sbatch=batch_size, lr=lr, input_size=784, hidden=800)
print('-'*100)

# Loop tasks
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
for t, ncla in taskcla:
    print('*'*100)
    print('Task {:2d} ({:s})'.format(t+1, data[t]['name']))
    print('*'*100)

    xtrain = data[t]['train']['x'].cuda()
    ytrain = data[t]['train']['y'].cuda()
    xvalid = data[t]['test']['x'].cuda()
    yvalid = data[t]['test']['y'].cuda()
    if t == 0:
        num_epochs = 35
        num = 0
    else:
        num_epochs = 50
        num = 1

    # num_epochs = 35
    # num = 1
    # Train
    print('batch_size={}, num_epochs={}, lr={}, distacne={}'.format(batch_size, num_epochs, lr, num))
    appr.train(t, xtrain, ytrain, xvalid, yvalid, data, lr, num_epochs=num_epochs, nums=num)
    print('-'*100)

    # Test
    print("Test on Previous Datasets:")
    for u in range(t+1):
        xtest = data[u]['test']['x'].cuda()
        ytest = data[u]['test']['y'].cuda()
        test_loss, test_acc = appr.eval(xtest, ytest)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.2f}% <<<'.format(u, data[u]['name'], test_loss, 100*test_acc))
        acc[t, u] = test_acc
        lss[t, u] = test_loss

xtest = data[2]['test']['x'].cuda()
ytest = data[2]['test']['y'].cuda()
test_loss, test_acc = appr.eval(xtest, ytest)
print('owm acc={:5.2f}% <<<'.format(100*test_acc))
acc[t, u] = test_acc

# Done
print('*'*100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t', end='')
    for j in range(acc.shape[1]):
        print('{:5.2f}% '.format(100*acc[i, j]), end='')
    print()
print('*'*100)
cur_time = datetime.now()
h, remainder = divmod((cur_time - prev_time).seconds, 3600)
m, s = divmod(remainder, 60)
time_str = "Time %02d:%02d:%02d" % (h, m, s)
print(time_str)
print('Done!')





