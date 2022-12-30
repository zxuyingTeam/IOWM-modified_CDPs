import argparse
import math
import os
import numpy as np
import scipy.io as scio
import time
from IOWMLayer_CDP_LCDP import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warning


# define the Hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, 
default="/home/lxt/working/workspace/IOWM_ultimate/celebA/data/")
parser.add_argument('--class_num', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--LCDP', type=bool, default=True)
parser.add_argument('--contextual_information', type=int, default=0,
                    help='0--embedding vector;1--orthogonal vector')
parser.add_argument('--lambda_loss', type=float, default=1e-3)
parser.add_argument('--train_context', type=bool, default=True)
parser.add_argument('--record_task', type=int, default=10, help='the split lines of tasks')
parser.add_argument('--threshold1', type=int, default=7, help='the first part implement iowm distance')
parser.add_argument('--threshold2', type=int, default=13)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

data_path = args.data_path
class_num = args.class_num
num_epochs = args.num_epochs
batch_size = args.batch_size
LCDP = args.LCDP
contextual_information = args.contextual_information
lambda_loss = args.lambda_loss
train_context = args.train_context
record_task = args.record_task
threshold1 = args.threshold1
threshold2 = args.threshold2
alpha_list = [0.15, 1.0, 0.1]

Path_All = data_path + 'celebA_mat50/'
Data_Path_train = os.path.join(Path_All, 'train')
Data_Path_val = os.path.join(Path_All, 'test')

hidden_size = 4000

OWM = OWMLayer_cdp([[2048, hidden_size], [hidden_size, class_num]], alpha=[1.0, ],
                   l2_reg_lambda=lambda_loss, train_context=train_context,
                   contextual_information=contextual_information, LCDP=LCDP,
                   args=args)

print('LCDP=', LCDP, ' | contextual_information=', contextual_information,
      ' | batch_size=', batch_size, ' | num_epochs=', num_epochs, ' | device=', args.device,
      ' | record_task=', record_task, ' | threshold1=', threshold1, ' | threshold2=', threshold2,
      ' | the hidden size is: ', hidden_size, ' | train_context is: ', train_context,
      ' | lambda_loss=', lambda_loss, ' | alpha_list=', alpha_list)

def my_test(class_begin=None, class_end=None):
    batch_size = 100
    acc_array = []
    for task_index in range(class_begin, class_end):
        correct = 0
        total = 0
        for num_index in range(10):
            mat_data = os.path.join(Data_Path_val, 'celebAdata' + str(num_index))
            test = scio.loadmat(mat_data)
            testimages = test['data']
            testlabels = test['lables'][:, task_index]
            test_length = len(testimages)  # 60
            for i in range(math.ceil(test_length / batch_size)):
                start = batch_size * i
                index_end = min(start + batch_size, test_length)
                batch_x = testimages[start:index_end, :]
                batch_y = testlabels[start:index_end]
                correct_each, total_each = OWM.owm_learn(batch_x, batch_y, train=False, task_index=task_index)
                total += total_each
                correct += correct_each
        test_acc = 100 * correct.float() / total
        print('the {:d}th task accuracy is: {:.2f}%'.format(task_index + 1, test_acc))
        acc_array.append(test_acc)
    return test_acc, acc_array

# train
def train(task_index, counts=0, idx=0, record_task=0, threshold1=0, threshold2=0):
    """
    :param task_index: the name of tasks
    :param counts: the number of batchs
    :param idx: have runed tasks number
    :return:
    """
    total = 0
    correct = 0
    for num_index in range(0, 92):
        mat_data = os.path.join(Data_Path_train, 'celebAdata' + str(num_index))
        train = scio.loadmat(mat_data)
        trainimages = train['data']
        trainlabels = train['lables']
        train_length = len(trainimages)
        ss = np.arange(train_length)
        np.random.shuffle(ss)
        images = trainimages[ss, :]
        labels = trainlabels[ss, task_index]
        for i in range(math.ceil(train_length / batch_size)):
            counts += 1
            if idx < record_task:  # the split line
                if counts < threshold1:
                    iowm = False
                else:
                    iowm = True
            else:
                if counts < threshold2:
                    iowm = False
                else:
                    iowm = True
            start = batch_size * i
            index_end = min(start + batch_size, train_length)
            batch_x = images[start:index_end, :]
            batch_y = labels[start:index_end]

            OWM.owm_learn(batch_x, batch_y, train=True, alpha_list=alpha_list, task_index=task_index, iowm=iowm)
            correct_each, total_each = OWM.owm_learn(batch_x, batch_y, train=False, task_index=task_index)
            total += total_each
            correct += correct_each
    print('Train Epoch_number:[{:d}/{:d}],curr_acc:{:.2f}%'.format(epoch + 1, num_epochs,
                              100 * correct.float() / total))
    return counts

Task_num = 40
start_time = time.time()

for seed_i in [4]:
    Task_array = np.arange(40)
    # if seed_i:
    seed = seed_i
    print('seed ', seed)
    np.random.seed(seed)
    np.random.shuffle(Task_array)
    task_j = 0
    idx = 0
    for task_index in Task_array[task_j:]:
        task_j += 1
        counts = 0
        print("Training owm_cdp:->>> [ {:d} / {:d} ] ... [ {:d} / {:d} ] ...".format(task_j, Task_num, task_index + 1,
                                                                                     Task_num))
        for epoch in range(num_epochs):
            total = 0
            correct = 0
            counts = train(task_index=task_index, counts=counts, idx=idx, record_task=record_task,
                           threshold1=threshold1, threshold2=threshold2)
        accu_all, _ = my_test(class_begin=task_index, class_end=task_index + 1)
    _, acc_array = my_test(class_begin=0, class_end=Task_num)
    print()
    print('All_acc:{:.2f} %'.format(np.mean(acc_array)))

end_time = time.time()
total_time = end_time - start_time
h, s = divmod(total_time, 3600)
d, h = divmod(h, 24)
s, m = divmod(s, 60)
time_str = "Time %02d:%02d:%02d:%02d" % (d, h, m, s)
print(time_str)


