import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import random
import argparse
import os
import sys

from model import UNet
from logger import Logger
from ACDC_dataset import load_dataset, CustomDataSet, my_transform1
from medpy.metric.binary import hd, dc, assd


def init_seed(seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def makedirs(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)  
    os.makedirs(os.path.join(exp_dir, "model"), exist_ok=True)  
    os.makedirs(os.path.join(exp_dir, "show_img"), exist_ok=True)  


def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)


def get_loader(dataset_root, batch_size):
    print('...Data loading is beginning...')

    train_data, val_data = load_dataset(dataset_root)
    imgs1 = {'train': train_data, 'val': val_data}

    dataset = {"train": CustomDataSet(data=imgs1["train"], transform=my_transform1), 
               "val": CustomDataSet(data=imgs1["val"], transform=None)}  
    shuffle = {'train': True, 'val': False}
    batch_size = {'train': batch_size, 'val': batch_size}

    dataloader = {x: DataLoader(dataset[x], 
        batch_size=batch_size[x], 
        shuffle=shuffle[x], 
        drop_last=True,
        num_workers=0) for x in ['train', 'val']
    }
    print('length of training set:', len(dataset['train']))
    print('length of validate set:', len(dataset['val']))
    print('===============================================')
    return dataloader


def train_UNET_model(opts, model, data_loaders, optimizer, Loss, logger, exp_dir, device):

    factor = opts.lr_factor
    best_acc = 0.0
    min_loss=100.0
    step = 0

    train_loss_history = []
    val_loss_history = []
    train_acc_history=[]
    val_acc_history=[]

    for epoch in range(opts.epochs):

        if (epoch+1) % opts.lr_step == 0:

            for p in optimizer.param_groups:
                p['lr'] *= factor
                print(p['lr'])
        print('Epoch {}/{}'.format(epoch+1, opts.epochs))
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for it, (img, seg, slice_label) in enumerate(data_loaders[phase]):
                if phase == 'train':
                    step += 1

                img, seg, slice_label = img.to(device), seg.long().to(device), slice_label.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    pred, _ = model(img)  
                    loss_seg = Loss(pred, seg)

                    regularization_loss = 0
                    for param in model.parameters(): 
                        regularization_loss = regularization_loss + torch.sum(torch.abs(param))

                    loss = loss_seg + regularization_loss*opts.weight_decay

                    prediction = np.uint8(np.argmax(pred.detach().cpu().numpy(), axis=1))  #10,224,224
                    acc = dc(prediction, seg.detach().cpu().numpy()) #dc1

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        train_loss_history.append(loss.cpu().detach().item())
                        train_acc_history.append(acc)

                    if phase == "val":
                        val_loss_history.append(loss.item())
                        val_acc_history.append(acc)

            if phase == "train":
                log_loss_summary(logger, train_loss_history, step, prefix='train loss')
                log_loss_summary(logger, train_acc_history, step, prefix='train acc')
                logger.scalar_summary("train loss", np.mean(train_loss_history), step) 
                train_loss_history = []
                train_acc_history = []

            if phase == "val":
                log_loss_summary(logger, val_loss_history, step, prefix='val loss')
                log_loss_summary(logger, val_acc_history, step, prefix='val acc')
                logger.scalar_summary("val loss", np.mean(val_loss_history), step) 

                if np.mean(val_acc_history) >= best_acc:
                    best_acc = np.mean(val_acc_history)
                    print('------saving model------')
                    torch.save(model.state_dict(), exp_dir+'/model/model.pth')
                    print("current saved model acc: {:4f}".format(best_acc))
                val_loss_history = []
                val_acc_history = []


def main(opts):
    print('========================')
    if opts.seed:
        print('initial fixed seed: ', opts.seed)
        init_seed(seed=opts.seed)
    else:
        print('random seed')
    exp_dir = "./experiments/" + str(opts.patients_num) + "/" + str(opts.model_type)
    makedirs(exp_dir)
    logger = Logger(os.path.join(exp_dir,"logs"))  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # setup dataloader
    dataset_root = "/data/zhangtw/project/self-surpe/ztw/data/ACDC/"+opts.data_root+"/data_2D_size_224_224_res_1.367_1.367.hdf5"
    loader = get_loader(dataset_root, opts.batch_size)  
    print('create dataset from: ', dataset_root)
    # set model
    model = UNet(n_channels=1, n_classes=4)
    if opts.pretrain:
        model_path = opts.model_path
        print('loading model from: ', model_path)
        pretrained_dict = torch.load(model_path, map_location='cuda:0')
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'final_' not in k}
        # 2. overwrite entries in the existingtate dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        print('training from scratch')
    print('========================')
    model.to(device)
    # optimizer
    params_to_update = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params_to_update, lr=opts.lr) 
    loss_seg = nn.CrossEntropyLoss()
    print('current experiment dir:', exp_dir)
    print('------Training------')
    train_UNET_model(opts, model, loader, optimizer, loss_seg, logger, exp_dir, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # exp save dir
    parser.add_argument("--model_type", type=str, default="task_ori", help="folder of current experiment")
    # dataset
    parser.add_argument("--data_root", type=str, default='data_V2', help="dataset version")
    parser.add_argument("--patients_num", type=int, default=4, help="number of training data")
    # model
    parser.add_argument("--pretrain", action="store_true") 
    parser.add_argument("--model_path", type=str, default='/data/zhangtw/project/self-surpe/ztw/pretrain/CMR/experiments/task_ori_1/model', help="different method")
    ##################
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs to train (default: 100)")
    parser.add_argument("--batch_size", type=int, default=10, help="input batch size for training (default: 16)")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate (default: 0.001)")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    ##################
    parser.add_argument("--lr_factor", type=float, default=0.5, help="lr scheduler factor")
    parser.add_argument("--lr_step", type=int, default=50, help="lr scheduler step")
    parser.add_argument("--weight_decay", type=float, default=0.00005, help="weight decay")

    opts = parser.parse_args()

    main(opts)
