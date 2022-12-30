import os
import scipy.io as scio
import numpy as np
from datetime import datetime
Path_All = '/mydata/CASIA_HWDB_new/data/CHW_mat3755/'

def make_train_each(start=0, end=3755):
    train_path = os.path.join(Path_All, 'train')
    train_path_each = os.path.join(Path_All, 'train_each')
    lens = len(os.listdir(train_path))
    for index in range(start, end):
        img = []
        labels = []
        for num_index in range(0, lens):
            mat_data = os.path.join(train_path, 'chwdata' + str(num_index))
            train = scio.loadmat(mat_data)
            trainimages = train['data']
            trainlabels = train['labels']
            trainlabels = trainlabels.squeeze()
            # print(trainlabels.shape)
            for i in range(len(trainimages)):
                if trainlabels[i] in range(index, index + 1):
                    img.append(trainimages[i])
                    labels.append(trainlabels[i])
        mat_name = 'chwdata' + str(index) + '.mat'
        os.makedirs(train_path_each, exist_ok=True)
        numpy_data = np.array(img)
        numpy_lables = np.array(labels)
        scio.savemat(os.path.join(train_path_each, mat_name),
                     mdict={'train_data_each': numpy_data, 'train_label_each': numpy_lables})

def make_test_each(start=0, end=3755):
    test_path = os.path.join(Path_All, 'test')
    test_path_each = os.path.join(Path_All, 'test_each')
    lens = len(os.listdir(test_path))
    for index in range(start, end):
        img = []
        labels = []
        for num_index in range(0, lens):
            mat_data = os.path.join(test_path, 'chwdata' + str(num_index))
            test = scio.loadmat(mat_data)
            testimages = test['data']
            testlabels = test['labels']
            testlabels = testlabels.squeeze()
            for i in range(len(testimages)):
                if testlabels[i] in range(index, index + 1):
                    img.append(testimages[i])
                    labels.append(testlabels[i])
        mat_name = 'chwdata' + str(index) + '.mat'
        os.makedirs(test_path_each, exist_ok=True)
        numpy_data = np.array(img)
        numpy_lables = np.array(labels)
        scio.savemat(os.path.join(test_path_each, mat_name),
                     mdict={'test_data_each': numpy_data, 'test_label_each': numpy_lables})


prev_time = datetime.now()

make_train_each(start=0, end=3755)

cur_time = datetime.now()
h, remainder = divmod((cur_time - prev_time).seconds, 3600)
m, s = divmod(remainder, 60)
time_str = "Time %02d:%02d:%02d" % (h, m, s)
print("make train datasets run time:" + time_str)

prev_time = datetime.now()

make_test_each(start=0, end=3755)

cur_time = datetime.now()
h, remainder = divmod((cur_time - prev_time).seconds, 3600)
m, s = divmod(remainder, 60)
time_str = "Time %02d:%02d:%02d" % (h, m, s)
print("make test datasets run time is:" + time_str)
