import os
import sys
import argparse

import numpy as np
import cv2

import prep_dataset.prep_SBD_dataset as prep_SBD_dataset
from modules.CASENet import CASENet_resnet101

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('model', type=str,
                    help="path to the checkpoint containing the trained weights")
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
        test_lst = [x.strip().split()[0] for x in f.readlines()]
        if args.image_dir!='':
            test_lst = [
                args.image_dir+x if os.path.isabs(x)
                else os.path.join(args.image_dir, x)
                for x in test_lst]
else:
    image_file = os.path.join(args.image_dir, args.image_file)
    if os.path.exists(image_file):
        test_lst = [os.path.join(args.image_dir, os.path.basename(image_file))]
    else:
        raise IOError('nothing to be tested!')

num_cls = 20
model = CASENet_resnet101(pretrained=True, num_classes=num_cls)
utils.load_pretrained_model(model, args.model)
crop_size = 512
for idx_img in xrange(len(test_lst)):
    in_ = cv2.imread(test_lst[idx_img]).astype(np.float32)
    width, height = in_.shape[1], in_.shape[0]
    if(crop_size < width or crop_size < height):
        raise ValueError('Input image size must be smaller than crop size!')
    pad_x = crop_size - width
    pad_y = crop_size - height
    in_ = cv2.copyMakeBorder(in_, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=mean_value)
    in_ -= np.array(mean_value)
    in_ = in_.transpose((2,0,1))    # HxWx3 -> 3xHxW
    in_ = in_[np.newaxis, ...]      # 3xHxW -> 1x3xHxW
    
    score_feats5, score_fuse_feats = model(in_)
    
    img_base_name = os.path.basename(test_lst[idx_img])
    img_result_name = os.path.splitext(img_base_name)[0]+'.png'
    for idx_cls in xrange(num_cls):
        score_pred = score_fuse_feats.data[0][idx_cls, 0:height, 0:width]
        im = (score_pred*255).astype(np.uint8)
        result_root = os.path.join(args.output_dir, 'class_'+str(idx_cls+1))
        if not os.path.exists(result_root):
            os.makedirs(result_root)
        cv2.imwrite(
            os.path.join(result_root, img_result_name),
            im)

    print 'processed: '+test_lst[idx_img]
    sys.stdout.flush()

print 'Done!'
