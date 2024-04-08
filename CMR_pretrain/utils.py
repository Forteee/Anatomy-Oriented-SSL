#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import random
import cv2
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from DSBCC_dataset import CustomDataSet


def get_loader(data_path, batch_size):
    dataset = {"train": CustomDataSet(data_path=data_path['train'], transform=None),
               "val": CustomDataSet(data_path=data_path['val'], transform=None, seed=1)}
    shuffle = {'train': True, 'val': False}
    drop_last = {'train': True, 'val': False}
    batch_size={'train':batch_size,'val':batch_size}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size[x],
                                shuffle=shuffle[x], num_workers=0, drop_last=drop_last[x]) for x in ['train', 'val']}
    return dataloader

def makedirs(exp_dir):
    os.makedirs(os.path.join(exp_dir,"logs"), exist_ok=True)  # log dir
    os.makedirs(os.path.join(exp_dir, "model"), exist_ok=True)  # model save folder
    os.makedirs(os.path.join(exp_dir, "heatmap_pre"), exist_ok=True)  # heatmap save folder
    
def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)

def init_seed(seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def show_heatmap_pre_multi(exp_dir, epoch, index, name, heatmap_pre):
    # path=osp.join(exp_dir, "heatmap_pre",str(epoch),".png")
    heatmap_show=heatmap_pre[:,0,:,:]+heatmap_pre[:,1,:,:]
    for i in range(int(heatmap_pre.size(0))):
        path = "./" + exp_dir + "/heatmap_pre/" + str(epoch) + "_" + str(index) + "_" + str(i) + "_" + name + ".jpg"
        heatmap_pre_show = cv2.normalize(heatmap_show[i].cpu().detach().numpy(), None, 0, 255, cv2.NORM_MINMAX)
        heatmap_pre_show = np.uint8(np.array(heatmap_pre_show.reshape(224,224)))
        cv2.imwrite(path, heatmap_pre_show)  # "NORM_MINMAX