import torch
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
        self.load_mode = load_mode
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.transform = transform
        self.mode = mode
        self.height = 512
        self.width = 512
        self.N = min(self.height, self.width)
        self.inputs, self.targets = [], []

        x = np.linspace(-self.height / 2 + 0.5, self.height / 2 - 0.5, self.height)
        y = np.linspace(-self.width / 2 + 0.5, self.width / 2 - 0.5, self.width)
        self.Ix, self.Iy = np.meshgrid(x, y)  

        #self.lists = ['L058', 'L064', 'L212', 'L150', 'L266', 'L019', 'L116', 'L193', 'L232', 'L186']
        self.lists = ['L058', 'L064', 'L212', 'L150', 'L266', 'L019', 'L116', 'L193', 'L232', 'L186', 'L248', 'L134',
                      'L273', 'L033', 'L077', 'L145', 'L114', 'L148', 'L220']
        if mode == 'train':
            input_ = [f for f in input_path if f.split('/')[-1].split('_')[0] in self.lists]
            target_ = [f for f in target_path if f.split('/')[-1].split('_')[0] in self.lists]
            # input_ = [f for f in input_path if test_patient not in f]
            # target_ = [f for f in target_path if test_patient not in f]
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
            input_img, target_img = np.fromfile(input_img, dtype=image_dtype), np.fromfile(target_img,
                                                                                           dtype=image_dtype)
            input_img = np.reshape(input_img, (self.height, self.width))
            target_img = np.reshape(target_img, (self.height, self.width))
            input_img = np.array(input_img, dtype=np.float32)
            target_img = np.array(target_img, dtype=np.float32)
            target_img[self.Ix ** 2 + self.Iy ** 2 > (self.N / 2 - 1) ** 2] = 0
            input_img = self.normal(input_img)
            target_img = self.normal(target_img)

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        return (input_img, target_img)

    def normal(self, image):
        if isinstance(image, np.ndarray):
            maxs, mins = np.max(image), np.min(image)
            image = (image - mins) / (maxs - mins)
        elif isinstance(image, torch.Tensor):
            maxs, mins = torch.max(image), torch.min(image)
            image = (image - mins) / (maxs - mins)
        return image

    def concatenate_alldata(self):
        for idx in range(len(self.target_)):
            input_img, target_img = self.input_[idx], self.target_[idx]
            input_img, target_img = np.fromfile(input_img, dtype=image_dtype), np.fromfile(target_img, dtype=image_dtype)
            input_img = np.reshape(input_img, (self.height, self.width))
            target_img = np.reshape(target_img, (self.height, self.width))
            target_img[self.Ix ** 2 + self.Iy ** 2 > (self.N / 2 - 1) ** 2] = 0
            input_img = self.normal(input_img)
            target_img = self.normal(target_img)
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


