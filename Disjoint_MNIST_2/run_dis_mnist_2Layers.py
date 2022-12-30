import sys, argparse
import numpy as np
import torch
import utils
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"  # use gpu0,1


print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  '+os.environ["CUDA_VISIBLE_DEVICES"])
print('='*100)
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

import owm_2Layers as approach
import linear_owm_2Layers as network

########################################################################################################################

# Load
print('Load data...')
data, taskcla, inputsize = utils.get()
print('Input size =', inputsize, '\nTask info =', taskcla)

# Inits
print('Inits...')
net = network.Net(784, 800, 10).cuda()

batch_size = 40
num_epochs = 35
lr = 0.2
print('batch_size={}, num_epochs={}, lr={}'.format(batch_size, num_epochs, lr))
appr = approach.Appr(net, nepochs=num_epochs, sbatch=batch_size, lr=lr)
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

    # Train
    appr.train(t, xtrain, ytrain, xvalid, yvalid, data)
    print('-'*100)

    # Test
    for u in range(t+1):
        xtest = data[u]['test']['x'].cuda()
        ytest = data[u]['test']['y'].cuda()
        test_loss, test_acc = appr.eval(xtest, ytest)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.2f}% <<<'.format(u, data[u]['name'], test_loss, 100*test_acc))
        acc[t, u] = test_acc
        lss[t, u] = test_loss


# Done
print('*'*100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t', end='')
    for j in range(acc.shape[1]):
        print('{:5.2f}% '.format(100*acc[i, j]), end='')
    print()
print('*'*100)
print('Done!')
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  '+os.environ["CUDA_VISIBLE_DEVICES"])
print('='*100)




