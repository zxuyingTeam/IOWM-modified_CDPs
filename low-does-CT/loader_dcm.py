import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset

image_dtype = np.float64
class datasets(Dataset):
    def __init__(self, mode, load_mode, saved_path, test_patient, patch_stride=None, patch_size=None, transform=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"
        assert load_mode in [0, 1], "load_mode is 0 or 1"

        input_path = sorted(glob(os.path.join(saved_path, '*_input.raw')))
        target_path = sorted(glob(os.path.join(saved_path, '*_target.raw')))
        # print('len(inputs)=', len(input_path))
        self.load_mode = load_mode
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.transform = transform
        self.mode = mode
        self.height = 512
        self.width = 512
        self.inputs, self.targets = [], []

        if mode == 'train':
            input_ = [f for f in input_path if test_patient not in f]
            target_ = [f for f in target_path if test_patient not in f]
            if load_mode == 0:  # batch data load
                self.input_ = input_
                self.target_ = target_
            else:  # all data load
                self.input_ = [np.fromfile(f, dtype=image_dtype) for f in input_]
                self.target_ = [np.fromfile(f, dtype=image_dtype) for f in target_]
        else:  # mode =='test'
            input_ = [f for f in input_path if test_patient in f]
            target_ = [f for f in target_path if test_patient in f]
            input_ = sorted(input_)
            target_ = sorted(target_)
            if load_mode == 0:
                self.input_ = input_
                self.target_ = target_
            else:
                self.input_ = [np.fromfile(f, dtype=image_dtype) for f in input_]
                self.target_ = [np.fromfile(f, dtype=image_dtype) for f in target_]

        if self.patch_size:
            self.concatenate_alldata()

    def __len__(self):
        if self.patch_size:
            return len(self.targets)
        return len(self.target_)

    def __getitem__(self, idx):
        if self.patch_size:
            input_img, target_img = self.inputs[idx], self.targets[idx]
            input_img = np.array(input_img, dtype=np.float32)
            target_img = np.array(target_img, dtype=np.float32)
            if self.transform:
                input_img = self.transform(input_img)
                target_img = self.transform(target_img)
                return (input_img, target_img)
            else:
                return (input_img, target_img)
        input_img, target_img = self.input_[idx], self.target_[idx]
        if self.load_mode == 0:
            input_img, target_img = np.fromfile(input_img, dtype=image_dtype), np.fromfile(target_img, dtype=image_dtype)
            input_img = np.reshape(input_img, (self.height, self.width))
            target_img = np.reshape(target_img, (self.height, self.width))
            input_img = np.array(input_img, dtype=np.float32)
            target_img = np.array(target_img, dtype=np.float32)

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        return (input_img, target_img)

    def concatenate_alldata(self):
        for idx in range(len(self.target_)):
            input_img, target_img = self.input_[idx], self.target_[idx]
            input_img, target_img = np.fromfile(input_img, dtype=image_dtype), np.fromfile(target_img, dtype=image_dtype)
            input_img = np.reshape(input_img, (self.height, self.width))
            target_img = np.reshape(target_img, (self.height, self.width))
            self.get_patch(input_img, target_img)

    def get_patch(self, full_input_img, full_target_img):
        # when the shape of input and target is not equal,system.exit(0)
        assert full_input_img.shape == full_target_img.shape
        h, w = full_input_img.shape
        for i in range(0, h, self.patch_stride):
            for j in range(0, w, self.patch_stride):
                patch_input_img = full_input_img[i:i + self.patch_size, j:j + self.patch_size]
                patch_target_img = full_target_img[i:i + self.patch_size, j:j + self.patch_size]
                self.inputs.append(patch_input_img)
                self.targets.append(patch_target_img)

