import cv2
import numpy as np
import nibabel as nib
import os
import os.path as osp


def may_make_dir(path):
  if path in [None, '']:
    return
  if not osp.exists(path):
    os.makedirs(path)

def convert_to_one_hot_batch(seg):
    res = np.zeros([seg.shape[0],4,seg.shape[1],seg.shape[2]], seg.dtype) 
    for i in range(len(seg)):
        res[i]=convert_to_one_hot(seg[i])
    return res

def convert_to_one_hot(seg):
    vals = np.unique(seg)
    vals=np.array([0,1,2,3])
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for c in range(len(vals)):
        res[c][seg == c] = 1
    return res


def resize_image_by_padding(image, new_shape, pad_value=None):
    shape = tuple(list(image.shape))
    new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2, len(shape))), axis=0))
    if pad_value is None:
        if len(shape) == 2:
            pad_value = image[0, 0]
        elif len(shape) == 3:
            pad_value = image[0, 0, 0]
        else:
            raise ValueError("Image must be either 2 or 3 dimensional")
    res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
    start = np.array(new_shape) / 2. - np.array(shape) / 2.
    if len(shape) == 2:
        res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1])] = image
    elif len(shape) == 3:
        res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1]),
        int(start[2]):int(start[2]) + int(shape[2])] = image
    return res

def get_square_crop(img, base_size, crop_size):

    res = img
    height, width = res.shape

    if height < base_size[0]:
            diff = base_size[0] - height
            extend_top = int(diff / 2)
            dx_odd= int(extend_top % 2 == 1)
            extend_bottom = int(diff - extend_top)
            res = cv2.copyMakeBorder(res, extend_top+dx_odd, extend_bottom, 0, 0, borderType=cv2.BORDER_CONSTANT, value=0)
            height = base_size[0]


    if width < base_size[1]:
            diff = base_size[1] - width
            extend_top = int(diff / 2)
            dy_odd = int(extend_top % 2 == 1)
            extend_bottom = int(diff - extend_top)
            res = cv2.copyMakeBorder(res, 0, 0, extend_top+dy_odd, extend_bottom, borderType=cv2.BORDER_CONSTANT, value=0)
            width = base_size[1]


    crop_y_start = (height - crop_size[0]) / 2
    crop_x_start = (width - crop_size[1]) / 2
    res = res[int(crop_y_start):int(crop_y_start + crop_size[0]), int(crop_x_start):int(crop_x_start + crop_size[1])]
    return res

def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False

def load_nii(img_path):

    '''
    Shortcut to load a nifti file
    '''

    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

def save_nii(img_path, data, affine, header):
    '''
    Shortcut to save a nifty file
    '''

    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)