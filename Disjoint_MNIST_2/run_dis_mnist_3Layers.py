import sys, argparse
import numpy as np
import torch
import utils
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu


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

import owm_3Layers as approach
import linear_owm_3Layers as network

########################################################################################################################

# Load
print('Load data...')
data, taskcla, inputsize = utils.get()
print('Input size =', inputsize, '\nTask info =', taskcla)

# Inits
print('Inits...')
net = network.Net(input_size=784, hidden1=800, hidden2=800, output_size=10, seed=79).cuda()

batch_size = 40
num_epochs = 20
lr = 0.2
appr = approach.Appr(net, nepochs=num_epochs, sbatch=batch_size, lr=lr, input_size=784, hidden1=800, hidden2=800)
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
    appr.train(t, xtrain, ytrain, xvalid, yvalid, data)
    print('-'*100)

    # Test
    print("Test on Previous Datasets:")
    for u in range(t+1):
        xtest = data[u]['test']['x'].cuda()
        ytest = data[u]['test']['y'].cuda()
        test_loss, test_acc = appr.eval(xtest, ytest)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.2f}% <<<'.format(u, data[u]['name'], test_loss, 100*test_acc))

print('*'*100)
print('Done!')
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  '+os.environ["CUDA_VISIBLE_DEVICES"])
print('='*100)




