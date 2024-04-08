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

from networks.premodel import  UNet
from networks.pre_model_res import  Resnet_Unet
from networks.premodel_Alex import  Alex_Unet
from networks.premodel_deeplab import  DeepLabV3

from DSBCC_dataset import CustomDataSet
from logger import Logger
from utils import get_loader, makedirs, log_loss_summary, init_seed, show_heatmap_pre_multi


# loss
def self_Lv_dis_loss(y_input, y_target):
    if abs(y_input-y_target)<=0.5:
        return 0
    else:
        loss_fn_lv = nn.MSELoss()
        loss_lv = loss_fn_lv(y_input, y_target)
        return loss_lv

class self_mseloss(nn.Module):

    def __init__(self):
        super(self_mseloss, self).__init__()

    def forward(self, y_input, y_target):
        num = torch.numel(y_input)
        mse_loss = torch.sum((y_input - y_target) ** 2)
        return mse_loss / num

#
def train(opts, model, dataloader, optimizer, loss_t1, loss_t2, alpha, beta, logger, device):
    '''
        loss_t1   loss for pretext task 1
        loss_t2   loss for pretext task 2
    '''
    exp_dir = opts.exp_dir
    weight_decay = opts.weight_decay
    loss_train = []
    Hloss_train = []
    Sloss_train = []
    loss_val = []
    Hloss_val = []
    Sloss_val = []
    min_loss= 10000  # loss upper bound
    step = 0
    for epoch in range(opts.epochs):
        # scheduler
        if epoch % 100 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.3
        print('Epoch {}/{}'.format(epoch+1, opts.epochs))
        print('-' * 10) 

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for it, (short_axis, heatmap_truex, slice_label, stack_id, time_id, sax_path) in enumerate(dataloader[phase]):
                if phase == "train":
                    step += 1

                short_axis, heatmap_truex, slice_label = short_axis.to(device), heatmap_truex.to(device), slice_label.to(device)
                with torch.set_grad_enabled(phase=='train'):
                    heatmap_pred, slice_pred = model(short_axis)  # [batch 2 224 224], [batch 1]
                    # loss for pretext task 1
                    loss_heat = loss_t1(heatmap_pred, heatmap_truex.float())
                    # loss for pretext task 2
                    loss_slice = loss_t2(slice_pred.view(-1), slice_label.view(-1).float())
                    # weight decay
                    regularization_loss = 0
                    for param in model.parameters():
                        regularization_loss = regularization_loss + torch.sum(torch.abs(param))
                    # total loss
                    loss = alpha*loss_heat + beta*loss_slice + weight_decay*regularization_loss
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # report
                        loss_train.append(loss.item())
                        Hloss_train.append(loss_heat.item())
                        Sloss_train.append(loss_slice.item())
                        if epoch % 10 == 0 and step % 25 == 0:
                            show_heatmap_pre_multi(exp_dir, epoch, step, 'train_x', heatmap_pre=heatmap_truex)
                            show_heatmap_pre_multi(exp_dir, epoch, step, 'train_pred', heatmap_pre=heatmap_pred)
                    elif phase == 'val':
                        # report
                        loss_val.append(loss.cpu().detach().item())
                        Hloss_val.append(loss_heat.cpu().detach().item())
                        Sloss_val.append(loss_slice.cpu().detach().item())
                        # 画出图像
                        if epoch % 10 == 0 and step % 10 == 0:
                            show_heatmap_pre_multi(exp_dir, epoch, step, 'test_x', heatmap_pre=heatmap_truex)
                            show_heatmap_pre_multi(exp_dir, epoch, step, 'test_pred',  heatmap_pre=heatmap_pred)
            # end current epoch train
            if phase == "train":
                log_loss_summary(logger, loss_train, step, prefix='train')
                log_loss_summary(logger, Hloss_train, step, prefix='train heatmap')
                log_loss_summary(logger, Sloss_train, step, prefix='train slice')
                logger.scalar_summary("train loss", np.mean(loss_train), step)
                loss_train = []
                Hloss_train = []
                Sloss_train = []

            if phase == "val":
                log_loss_summary(logger, loss_val, step, prefix='val')
                log_loss_summary(logger, Hloss_val, step, prefix='val heatmap')
                log_loss_summary(logger, Sloss_val, step, prefix='val slice')
                logger.scalar_summary("val loss", np.mean(loss_val), step) 
                # save model 
                if np.mean(loss_val) < min_loss:
                    min_loss = np.mean(loss_val)
                    print('------saving model------')
                    torch.save(model.state_dict(), exp_dir+'/model/model.pth')
                    print("current saved model loss: {:4f}".format(min_loss))
                loss_val = []
                Hloss_val = []
                Sloss_val = []


def main(opts):
    # setup experiment setting
    if opts.seed:
        print("seed: ", opts.seed)
        init_seed(seed=opts.seed)
    else:
        print('no seed')
    makedirs(opts.exp_dir)  
    logger = Logger(os.path.join(opts.exp_dir,"logs"))  
    # setup model and dataloader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_paths = {'train': os.path.join(opts.root_path, 'train'), 'val': os.path.join(opts.root_path, 'test')}
    loader = get_loader(data_paths, opts.batch_size)
    ## model
    if opts.model == 'unet':
        model = UNet()

    elif opts.model == 'resnet':
        model = Resnet_Unet()

    elif opts.model == 'alexnet':
        model = Alex_Unet()

    elif opts.model == 'deeplab':
        model = DeepLabV3(class_num=2)

    if opts.pretrain:
        pretrained_dict = torch.load(pretrain_path)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existingtate dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    model.to(device)
    ## optimizer
    params_to_update = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params_to_update, opts.lr)

    loss_t1 = self_mseloss()
    loss_t2 = nn.MSELoss()
    alpha = opts.alpha
    beta = opts.beta
    # train model
    print('------Training------')
    train(opts, model, loader, optimizer, loss_t1, loss_t2, alpha, beta, logger, device)


if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./experiments/test1", help="folder of current experiment")
    parser.add_argument("--root_path", type=str, default="./data_DSBCC/DSBCC_processed/", help="folder of current experiment")
    parser.add_argument("--model", type=str, default="unet", help="network structures")
    parser.add_argument("--pretrain", action="store_true")  
    # 
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs to train (default: 100)")
    parser.add_argument("--batch_size", type=int, default=20, help="input batch size for training (default: 16)")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate (default: 0.001)")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    #
    parser.add_argument("--alpha", type=int, default=1, help="pertrain task 1")
    parser.add_argument("--beta", type=int, default=1, help="pretrain task 2")
    parser.add_argument("--weight_decay", type=float, default=0.00000, help="weight decay")
    #
    opts = parser.parse_args()

    main(opts)



