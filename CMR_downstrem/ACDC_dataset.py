from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2
import os
from sklearn.model_selection import KFold
import pickle
#from ACDC.utils.util import get_square_crop, resize_image_by_padding
import torchvision.transforms.functional as tf
import random
import h5py
# from ACDC.predict_acdc import image_utils

class Augmentation:
    def __init__(self):
        pass

    def randomResizeCrop(self, img, mask, p=1.0, scale=(0.7, 0.8),
                         ratio=(1, 1)):  
        i = random.random()
        if i > (1 - p):
            i, j, h, w = transforms.RandomResizedCrop.get_params(mask, scale=scale, ratio=ratio)
            img = tf.crop(img, i, j, h, w)
            mask = tf.crop(mask, i, j, h, w)
        return img, mask

    def scaling(self, image, mask, p=1.0):
        i = random.random()
        if i > (1 - p):
            h, w = image.size
            image = tf.crop(image, 0, 0, 220, 220)
            image = tf.resize(image, (w, h), Image.LANCZOS)

            mask = tf.crop(mask, 0, 0, 220, 220)
            mask = tf.resize(mask, (w, h), Image.LANCZOS)
        return image, mask

    def randomCrop(self, img, mask):
        i, j, h, w = transforms.RandomCrop.get_params(img, (224, 224))
        img = tf.crop(img, i, j, h, w)
        mask = tf.crop(mask, i, j, h, w)
        return img, mask

    def rotate(self, image, mask, p=1.0, angle=None):
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        i = random.random()
        if i > (1 - p):
            if angle == None:
                angle = transforms.RandomRotation.get_params([-15, 15]) 
            if isinstance(angle, list):
                angle = random.choice(angle)
            image = image.rotate(angle)
            mask = mask.rotate(angle)
        return image, mask

    def RandomChoice(self, image, mask, p=1.0):
        list = ["rotate", "randomResizeCrop"]
        trans = random.sample(list, 1)
        if trans[0] == "rotate":
            image, mask = self.rotate(image, mask, p)
        if trans[0] == "randomResizeCrop":
            image, mask = self.randomResizeCrop(image, mask, p)
        return image, mask


def my_transform1(image, mask):
    aug = Augmentation()#transform
    # aug=augmentation_function()#cv2
    image, mask = aug.rotate(image, mask, p=1.0)
    # image, mask = aug.randomCrop(image, mask)
    # image, mask=aug.randomResizeCrop(image, mask,p=1.0)
    # image, mask = aug.scaling(image, mask,p=0.2)
    # image, mask = aug.RandomChoice(image, mask)

    image = np.asarray(image)
    mask = np.asarray(mask)
    return image, mask

def load_dataset(root_dir):
    data= h5py.File(root_dir, 'r')
    images_train = data['images_train']
    labels_train = data['masks_train']
    slice_labels_train = data['slice_train']

    images_val = data['images_test']
    labels_val = data['masks_test']
    slcie_labels_val = data['slice_test']

    train_data = {}
    val_data = {}
    for i in range(len(images_train)):
        train_data[i] = {}
        train_data[i]['image'] = images_train[i]
        train_data[i]['mask'] = labels_train[i]
        train_data[i]['slice label'] = slice_labels_train[i]
    for i in range(len(images_val)):
        val_data[i] = {}
        val_data[i]['image'] = images_val[i]
        val_data[i]['mask'] = labels_val[i]
        val_data[i]['slice label'] = slcie_labels_val[i]
    return train_data, val_data

class CustomDataSet(Dataset):
    def __init__(
            self,
            data,
            transform=None
    ):
        self.transform = transform
        self.data = data


    def __getitem__(self, index):
        img = self.data[index]["image"]
        seg = self.data[index]["mask"]
        slice_label = self.data[index]["slice label"]
        if self.transform:
            img, seg = self.transform(img, seg)

        img = img.reshape(1, img.shape[0], img.shape[1])
        img = torch.from_numpy(img)
        seg = torch.from_numpy(seg)

        return img, seg, slice_label

    def __len__(self):
        count = len(self.data)
        return count