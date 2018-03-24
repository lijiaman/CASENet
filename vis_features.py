import os
import sys
import argparse

import numpy as np
import cv2
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_sbd_class_names():
    return[
        'Airplane',
        'Bicycle',
        'Bird',
        'Boat',
        'Bottle',
        'Bus',
        'Car',
        'Cat',
        'Chair',
        'Cow',
        'Diningtable',
        'Dog',
        'Horse',
        'Motorbike',
        'Person',
        'Pottedplant',
        'Sheep',
        'Sofa',
        'Train',
        'Tvmonitor'
    ]

def normalized_feature_map(fmap):
    fmap_min = fmap.min()
    fmap_max = fmap.max()
    fmap = (fmap-fmap_min)/(fmap_max-fmap_min)
    return fmap

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

# load net
num_cls = 20
model = CASENet_resnet101(pretrained=False, num_classes=num_cls)
utils.load_pretrained_model(model, args.model)
cls_names = get_sbd_class_names()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

input_size = 352
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

img_transform = transforms.Compose([
                transforms.Resize([input_size, input_size]),
                transforms.ToTensor(),
                normalize,
                ]))

for idx_img in xrange(len(test_lst)):
    img = Image.open(test_lst[idx_img]).convert('RGB')
    processed_img = img_transform(img).transpose(1,2)    
    
    score_feats1, score_feats2, score_feats3, score_feats5, score_fuse_feats = model.forward_for_vis(processed_img)

    img_base_name_noext = os.path.splitext(os.path.basename(test_lst[idx_img]))[0]
   
    score_feats_list = [score_feats1, score_feats2, score_feats3]
    score_feats_str_list = ['feats1', 'feats2', 'feats3'] 

    # vis side edge activation
    for i in xrange(len(score_feats_list)):
        feature = score_feats_list[i]
        feature_str = score_feats_str_list[i]

        side = normalized_feature_map(feature.data[0][0, :, :].cpu().numpy())
        im = (side*255).astype(np.uint8)
        cv.imwrite(
            os.path.join(args.output_dir, img_base_name_noext+'_'+feature_str+'.png'),
            im)
    # vis side class activation
    side_cls = normalized_feature_map(np.transpose(score_feats5.data[0].cpu().numpy(), (1, 2, 0)))
    for idx_cls in xrange(num_cls):
        side_cls_i = side_cls[:, :, idx_cls]
        im = (side_cls_i * 255).astype(np.uint8)
        cv.imwrite(
            os.path.join(args.output_dir, img_base_name_noext+'_'+'feats5'+'_'+cls_names[num_cls-idx_cls-1]+'.png'),
            im)

    print 'processed: '+test_lst[idx_img]
    sys.stdout.flush()

print 'Done!'
