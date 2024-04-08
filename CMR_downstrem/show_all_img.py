import h5py
import numpy as np
import torch.nn as nn
import torch
import cv2
import matplotlib.pyplot as plt
from ACDC.utils import util
from PIL import Image
import os
import glob
import nibabel as nib
import gc
import h5py
from skimage import transform


def  show_img(img, path,string, index):
    path = path +string+ "_"+str(index )+ ".jpg"
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure()
    plt.imshow(img , cmap=plt.cm.gray)
    # plt.imshow(heatmap_show1*1, alpha=0.5, cmap=plt.cm.viridis)
    plt.savefig(path)
    # plt.show()


if __name__ '__main__':
    input_folder='/data/zmm/Cardiac/data/ACDC/training'
    path_t="/data/zmm/Cardiac/code/my-cardiac/ACDC/my_exp_test/show_img/"
    for folder in os.listdir(input_folder):

        folder_path = os.path.join(input_folder, folder)

        if os.path.isdir(folder_path):

            for file in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):
                nifty_img =util.load_nii(file)
                for i in range(nifty_img[0].shape[2]):
                    img=np.squeeze(nifty_img[0][:, :, i])
                    show_img(img, path_t,  folder_path[-3:],i)


