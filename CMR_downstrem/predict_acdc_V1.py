import os
import glob
import numpy as np
import logging
import torch
import argparse
from skimage import transform
import time

import os.path as osp
#from ACDC.utils.util import may_make_dir
#from ACDC.utils.show import show_img, show_seg, show_seg1, show_seg2
from model import UNet
from predict_acdc import image_utils, metrics_acdc, util


# "https://github.com/baumgach/acdc_segmenter/blob/master/evaluate_patients.py"
def score_data(input_folder, output_folder, model_path, do_postprocessing=False, gt_exists=True, evaluate_all=False, use_iter=None):

    model = UNet(n_channels=1, n_classes=4)
    pretrained_dict = torch.load(model_path,  map_location='cuda:0')
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existingtate dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.cuda()

    target_resolution = (1.367, 1.367)
    image_size = (224, 224)
    nx, ny = image_size[:2]
    batch_size = 1
    num_channels = 4

    for folder in os.listdir(input_folder):

        folder_path = os.path.join(input_folder, folder)

        if os.path.isdir(folder_path):
            # if evaluate_all:
            #     train_test = 'test'  # always test
            # elif int(folder[-3:]) > 80:
            #     train_test = 'test'
            # else:
            #     train_test = 'train'


            if  evaluate_all:
                train_test = 'test'  # always test
            elif int(folder[-3:]) <= 1:
                train_test = 'train'

            elif int(folder[-3:])>64 and int(folder[-3:])<=84:  # and int(folder[-3:])>64 :
                train_test = 'test'
            else:
                train_test=""

            if train_test == 'test':

                infos = {}
                for line in open(os.path.join(folder_path, 'Info.cfg')):
                    label, value = line.split(':')
                    infos[label] = value.rstrip('\n').lstrip(' ')

                patient_id = folder.lstrip('patient')
                ED_frame = int(infos['ED'])
                ES_frame = int(infos['ES'])

                for file in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):
                    logging.info(' ----- Doing image: -------------------------')
                    logging.info('Doing: %s' % file)
                    logging.info(' --------------------------------------------')

                    file_base = file.split('.nii.gz')[0]

                    frame = int(file_base.split('frame')[-1])
                    img_dat = util.load_nii(file)
                    img = img_dat[0].copy()
                    img = image_utils.normalise_image(img)  # z-score

                    if gt_exists:
                        file_mask = file_base + '_gt.nii.gz'
                        mask_dat = util.load_nii(file_mask)
                        mask = mask_dat[0]

                    pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
                    scale_vector = (pixel_size[0] / target_resolution[0],
                                    pixel_size[1] / target_resolution[1])

                    predictions = []

                    for zz in range(img.shape[2]):
                        slice_img = np.squeeze(img[:,:,zz])
                        slice_rescaled = transform.rescale(slice_img,
                                                        scale_vector,
                                                        order=1,
                                                        preserve_range=True,
                                                        multichannel=False,
                                                        anti_aliasing=True,
                                                        mode='constant')

                        x, y = slice_rescaled.shape

                        x_s = (x - nx) // 2
                        y_s = (y - ny) // 2
                        x_c = (nx - x) // 2
                        y_c = (ny - y) // 2

                        # Crop section of image for prediction
                        if x > nx and y > ny:

                            slice_cropped = slice_rescaled[x_s:x_s+nx, y_s:y_s+ny]
                        else:
                            slice_cropped = np.zeros((nx,ny))
                            if x <= nx and y > ny:
                                slice_cropped[x_c:x_c+ x, :] = slice_rescaled[:,y_s:y_s + ny]
                            elif x > nx and y <= ny:
                                slice_cropped[:, y_c:y_c + y] = slice_rescaled[x_s:x_s + nx, :]
                            else:
                                slice_cropped[x_c:x_c+x, y_c:y_c + y] = slice_rescaled[:, :]

                        # GET PREDICTION
                        network_input = np.float32(np.tile(np.reshape(slice_cropped, (1, nx, ny)), (batch_size, 1, 1, 1)))
                        network_input = torch.from_numpy(network_input)
                        network_input = network_input.cuda()
                        
                        logits_out, slice_label = model(network_input)
                        logits_out = torch.softmax(logits_out, dim=1)
                        prediction_cropped = np.squeeze(logits_out.detach().cpu().numpy()[0, ...])  

                        # softmax = torch.nn.Softmax()
                        # pref = softmax(logits_out, dim=0)  
                        # mask_out = torch.argmax(pref, dim=1)

                        prediction_cropped = np.transpose(prediction_cropped,(1,2,0))
                        # ASSEMBLE BACK THE SLICES
                        slice_predictions = np.zeros((x, y, num_channels))
                        # insert cropped region into original image again
                        if x > nx and y > ny:
                            slice_predictions[x_s:x_s + nx, y_s:y_s + ny, :] = prediction_cropped
                        else:
                            if x <= nx and y > ny:
                                slice_predictions[:, y_s:y_s + ny, :] = prediction_cropped[x_c:x_c + x, :, :]
                            elif x > nx and y <= ny:
                                slice_predictions[x_s:x_s + nx, :, :] = prediction_cropped[:, y_c:y_c + y, :]
                            else:
                                slice_predictions[:, :, :] = prediction_cropped[x_c:x_c + x, y_c:y_c + y, :]

                        # RESCALING ON THE LOGITS
                        if gt_exists:
                            prediction = transform.resize(slice_predictions,
                                                          (mask.shape[0], mask.shape[1], num_channels),
                                                          order=1,
                                                          preserve_range=True,
                                                          anti_aliasing=True,
                                                          mode='constant')
                        else:  # This can occasionally lead to wrong volume size, therefore if gt_exists
                               # we use the gt mask size for resizing.
                            prediction = transform.rescale(slice_predictions,
                                                           (1.0/scale_vector[0], 1.0/scale_vector[1],1),
                                                           order=1,
                                                           preserve_range=True,
                                                           multichannel=False,
                                                           anti_aliasing=True,
                                                           mode='constant')

                        prediction = np.uint8(np.argmax(prediction, axis=-1))  

                    prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1,2,0)) 

                    # This is the same for 2D and 3D again
                    if do_postprocessing:
                        prediction_arr = image_utils.keep_largest_connected_components(prediction_arr) 
                    # may_make_dir(osp.join(base_path, "show_img"))
                    # show_seg2(base_path, int(0), 0, np.transpose(prediction_arr, (2, 1, 0)),
                    #           np.transpose(img, (2, 1, 0)))

                    if frame == ED_frame:
                        frame_suffix = '_ED'
                    elif frame == ES_frame:
                        frame_suffix = '_ES'
                    else:
                        raise ValueError('Frame doesnt correspond to ED or ES. frame = %d, ED = %d, ES = %d' %
                                         (frame, ED_frame, ES_frame))

                    # Save prediced mask
                    out_file_name = os.path.join(output_folder, 'prediction',
                                                 'patient' + patient_id + frame_suffix + '.nii.gz')
                    if gt_exists:
                        out_affine = mask_dat[1]
                        out_header = mask_dat[2]
                    else:
                        out_affine = img_dat[1]
                        out_header = img_dat[2]

                    logging.info('saving to: %s' % out_file_name)
                    util.save_nii(out_file_name, prediction_arr, out_affine, out_header)

                    # Save image data to the same folder for convenience
                    image_file_name = os.path.join(output_folder, 'image',
                                            'patient' + patient_id + frame_suffix + '.nii.gz')
                    logging.info('saving to: %s' % image_file_name)
                    util.save_nii(image_file_name, img_dat[0], out_affine, out_header)

                    if gt_exists:
                        # Save GT image
                        gt_file_name = os.path.join(output_folder, 'ground_truth', 'patient' + patient_id + frame_suffix + '.nii.gz')
                        logging.info('saving to: %s' % gt_file_name)
                        util.save_nii(gt_file_name, mask, out_affine, out_header)

                        # Save difference mask between predictions and ground truth
                        difference_mask = np.where(np.abs(prediction_arr-mask) > 0, [1], [0]) 
                        difference_mask = np.asarray(difference_mask, dtype=np.uint8)
                        diff_file_name = os.path.join(output_folder,
                                                      'difference',
                                                      'patient' + patient_id + frame_suffix + '.nii.gz')
                        logging.info('saving to: %s' % diff_file_name)
                        util.save_nii(diff_file_name, difference_mask, out_affine, out_header)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script to evaluate a neural network model on the ACDC challenge data")
    #parser.add_argument("EXP_PATH", type=str, help="Path to experiment folder (assuming you are in the working directory)")
    parser.add_argument('-t', '--evaluate_test_set', action='store_true')
    parser.add_argument('-a', '--evaluate_all', action='store_true')
    parser.add_argument('-i', '--iter', type=int, help='which iteration to use')
    parser.add_argument('--exp_path', type=str, help='exp path')
    args = parser.parse_args()

    evaluate_test_set = args.evaluate_test_set
    evaluate_all = args.evaluate_all

    if evaluate_test_set and evaluate_all:
        raise ValueError('evaluate_all and evaluate_test_set cannot be chosen together!')

    use_iter = args.iter
    if use_iter:
        logging.info('Using iteration: %d' % use_iter)

    # experiment path
    base_path = args.exp_path
    print('current experiment path: ', base_path)
    model_path = os.path.join(base_path, "model/model.pth")
    #model_path = os.path.join(base_path, "modal/ckpt.path")

    if evaluate_test_set:
        logging.warning('EVALUATING ON TEST SET')
        input_path = "./data_ACDC/training/"
        output_path = os.path.join(base_path, 'predictions_testset')
    elif evaluate_all:
        logging.warning('EVALUATING ON ALL TRAINING DATA')
        input_path = "./data_ACDC/training/"
        output_path = os.path.join(base_path, 'predictions_alltrain')
    else:
        logging.warning('EVALUATING ON VALIDATION SET')
        input_path = "./data_ACDC/training/"
        output_path = os.path.join(base_path, 'predictions')

    path_pred = os.path.join(output_path, 'prediction') 
    path_image = os.path.join(output_path, 'image')
    util.makefolder(path_pred)
    util.makefolder(path_image)

    if not evaluate_test_set:
        path_gt = os.path.join(output_path, 'ground_truth')  
        path_diff = os.path.join(output_path, 'difference')
        path_eval = os.path.join(output_path, 'results summary')  

        util.makefolder(path_diff)
        util.makefolder(path_gt)

    init_iteration = score_data(input_path,
                                output_path,
                                model_path,
                                do_postprocessing=True,
                                gt_exists=(not evaluate_test_set), 
                                evaluate_all=evaluate_all,  
                                use_iter=use_iter,)
    # output_path=base_path+"predictions/"
    # path_gt = os.path.join(output_path, 'ground_truth')
    # path_diff = os.path.join(output_path, 'difference')
    # path_eval = os.path.join(output_path, 'eval')
    if not evaluate_test_set:
        metrics_acdc.main(path_gt, path_pred, path_eval)