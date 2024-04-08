from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as tf
import torchvision.transforms as transforms
from PIL import Image
import torch
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import math
import numpy as np
import joblib
import math
from glob import glob
import cv2
import os
import random
from PIL import Image
import torch
import math


class Augmentation:
    def __init__(self):
        pass

    def randomResizeCrop(self, img, mask, p=1.0, scale=(0.7, 0.8), ratio=(1, 1)):
        i = random.random()
        if i > (1 - p):
            h,w=img.size
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)
            w1 = int(round(math.sqrt(target_area * aspect_ratio)))
            h1 = int(round(math.sqrt(target_area / aspect_ratio)))
            img=tf.resize(img, (w1, h1), Image.ANTIALIAS)
            mask = tf.resize(mask, (w1, h1), Image.ANTIALIAS)
            i = random.randint(0, w1-w )
            j = random.randint(0,h1-h )
            i, j, h, w = transforms.RandomResizedCrop.get_params(mask, scale=scale, ratio=ratio)
            img = tf.crop(img, i, j, h, w)
            mask = tf.crop(mask, i, j, h, w)
        return img, mask

    def scaling(self, x, y, p=1.0):
        image= x.copy()
        mask=y.copy()

        h, w = image.shape
        radio= random.uniform(0.92, 1.1 )
        w1=h1=int(round(math.sqrt(h*w * radio)))

        if random.random() < p:
            image = Image.fromarray(image)

            if radio<1.0:
                i0 = random.randint(0, w - w1)
                j0 = random.randint(0, w - w1)
                image = tf.crop(image, i0, j0, h1, w1)
            image = tf.resize(image, (w, h), Image.LANCZOS)
            image = np.asarray(image, dtype=np.float32)

            #heatmap
            for i in range(mask.shape[0]):
                heatmap = mask[i, :, :]
                heatmap = Image.fromarray(heatmap)
                if radio<1.0:
                    heatmap =tf.crop(heatmap, i0, j0, h1, w1)
                heatmap = tf.resize(heatmap, (w, h), Image.LANCZOS)
                mask[i, :, :] = heatmap
            heatmap_true = np.vstack((mask[0, :, :].reshape(1, h, w), mask[1, :, :].reshape(1, h, w)))
            mask = np.asarray(heatmap_true, dtype=np.float32)

        return image.reshape(1,h,w), mask

    def rotate(self, image, mask, p=0.5, angle=None):
        x = image.copy()
        y = mask.copy()
        h,w=x.shape
        cnt = 1
        while random.random() < p and cnt > 0:
            cnt = cnt - 1
            x = Image.fromarray(x)
            if angle == None:
                angle = transforms.RandomRotation.get_params([-3, 3])  # 随机选一个角度旋转
            if isinstance(angle, list):
                angle = random.choice(angle)

            x = x.rotate(angle)
            x = np.asarray(x, dtype=np.float32)

            # heatmap
            for i in range(y.shape[0]):
                heatmap= y[ i, :, :]
                heatmap = Image.fromarray(heatmap)
                y[i, :, :]=  np.asarray(heatmap.rotate(angle),dtype=np.float32)
            # heatmap_true = np.vstack(( y[0, :, :].reshape(1,h,w),  y[1, :, :].reshape(1,h,w)))
            # y = np.asarray(heatmap_true, dtype=np.float32)

        return x.reshape(1,h,w), y

    def RandomApply(self, image, mask, p=1.0):
        list = ["rotate", "scaling"]
        trans = random.sample(list, 1)
        if trans[0] == "rotate":
            image, mask = self.rotate(image, mask, p)
        if trans[0] == "scaling":
            image, mask = self.scaling(image, mask, p)
        return image, mask

    def RandomOrder(self, image, mask,p):
        if random.random() < 0.5:
            image, mask = self.rotate(image, mask, p)
            image, mask = self.scaling(image.reshape(image.shape[1],image.shape[2]), mask, p)
        else:
           image, mask = self.scaling(image, mask, p)
           image, mask = self.rotate(image.reshape(image.shape[1],image.shape[2]), mask, p)

        return image, mask


def my_transform1(image, mask):
    aug = Augmentation()
    image, mask = aug.RandomOrder(image.reshape(image.shape[1],image.shape[2]),mask, p=0.5)
    return image, mask


class CustomDataSet(Dataset):
    def __init__(self, data_path, transform=None, seed=None):
        self.data_path = data_path
        self.transform = transform
        self.pid_list = os.listdir(self.data_path)
        self.pid_list.sort(key = lambda x: int(x.split('.')[0]))
        self.seed = seed

    def __getitem__(self, index):
        case_path = os.path.join(self.data_path, self.pid_list[index])
        with  open(case_path, 'rb') as f:
            dicom_data = joblib.load(f)
        self.dicom_data = dicom_data

        if self.seed: 
            np.random.seed(self.seed+index)

        stack_num = len(dicom_data)
        stack_id = np.random.choice(stack_num)
        #
        time_num = len(dicom_data[stack_id]['image'])
        time_id = np.random.choice(time_num)

        short_axis = dicom_data[stack_id]['image'][time_id]
        heatmap_truex = dicom_data[stack_id]['heatmap']

        rel_gap = dicom_data[stack_id]['rel_gap']
        abs_gap = dicom_data[stack_id]['abs_gap']
        GT_slice = rel_gap / abs_gap

        sax_path = dicom_data[stack_id]['image_path'][time_id]

        if self.transform:
            short_axis, heatmap_truex = self.transform(short_axis, heatmap_truex)

        short_axis = torch.from_numpy(short_axis).float()
        heatmap_truex = torch.from_numpy(heatmap_truex).float()

        return short_axis, heatmap_truex, GT_slice, stack_id, time_id, sax_path

    def __len__(self):
        count = len(self.pid_list)

        return count
