import os
import sys
import argparse

import numpy as np
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import zipfile
import shutil
import h5py
from scipy.misc import imsave

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from modules.CASENet import CASENet_resnet101
from prep_dataset.prep_SBD_dataset import RGB2BGR
from prep_dataset.prep_SBD_dataset import ToTorchFormatTensor

import utils.utils as utils

def shifting(bitlist):
    """
    From https://stackoverflow.com/questions/12461361/bits-list-to-integer-in-python
    Convert a binary list to int
    """
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('-m', '--model', type=str,
                        help="path to the caffemodel containing the trained weights")
    parser.add_argument('-l', '--image_list', type=str, default='',
                        help="list of image files to be tested")
    parser.add_argument('-f', '--image_file', type=str, default='',
                        help="a single image file to be tested")
    parser.add_argument('-d', '--image_dir', type=str, default='',
                        help="root folder of the image files in the list or the single image file")
    parser.add_argument('-o', '--output_dir', type=str, default='.',
                        help="folder to store the test results")
    args = parser.parse_args(sys.argv[1:])
    
    # load input path
    if os.path.exists(args.image_list):
        with open(args.image_list) as f:
            ori_test_lst = [x.strip().split()[0] for x in f.readlines()]
            if args.image_dir!='':
                test_lst = [
                    args.image_dir+x if os.path.isabs(x)
                    else os.path.join(args.image_dir, x)
                    for x in ori_test_lst]
    else:
        image_file = os.path.join(args.image_dir, args.image_file)
        if os.path.exists(image_file):
            ori_test_list = [args.image_file]
            test_lst = [image_file]
        else:
            raise IOError('nothing to be tested!')
    
    # load net
    num_cls = 20
    model = CASENet_resnet101(pretrained=False, num_classes=num_cls)
    model = model.cuda()
    model = model.eval()
    cudnn.benchmark = True
    utils.load_pretrained_model(model, args.model)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Define normalization for data    
    normalize = transforms.Normalize(mean=[104.008, 116.669, 122.675], std=[1, 1, 1])
    
    img_transform = transforms.Compose([
                    transforms.Resize([352, 352]),
                    RGB2BGR(roll=True),
                    ToTorchFormatTensor(div=False),
                    normalize,
                    ])
    
    for idx_img in xrange(len(test_lst)):
        img = Image.open(test_lst[idx_img]).convert('RGB')
        processed_img = img_transform(img).unsqueeze(0)
        processed_img = utils.check_gpu(0, processed_img)    
        score_feats5, score_fuse_feats = model(processed_img) # 1 X 20 X 352 X 352
        
        score_output = F.sigmoid(score_fuse_feats.transpose(1,3).transpose(1,2)).squeeze(0) # H X W X 20
        score_pred_flag = (score_output>0.5)
        # Convert binary prediction to uint32
        im_arr = np.empty((score_fuse_feats.size()[2], score_fuse_feats.size()[3]), np.uint32)
        for i in xrange(score_fuse_feats.size()[2]):
            for j in xrange(score_fuse_feats.size()[3]):
                print("ori_binary list:{0}".format(list(score_pred_flag[i, j, :].data.cpu().numpy())))
                im_arr[i, j] = np.uint32(shifting(list(score_pred_flag[i, j].data.cpu().numpy().astype(np.int))))
                print("value:{0}".format(np.uint32(shifting(list(score_pred_flag[i, j].squeeze(0).data.cpu().numpy().astype(np.int))))))
                print("binary value:{0}".format(bin(np.uint32(shifting(list(score_pred_flag[i, j].squeeze(0).data.cpu().numpy().astype(np.int)))))))
        # Store value into img
        img_base_name_noext = os.path.splitext(os.path.basename(test_lst[idx_img]))[0]
        imsave(os.path.join(args.output_dir, img_base_name_noext+'.bmp'), im_arr)
        print 'processed: '+test_lst[idx_img]
    
    print 'Done!'

