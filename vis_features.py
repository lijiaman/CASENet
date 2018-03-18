import os
import sys
import argparse

import numpy as np
import cv2

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
parser.add_argument('model', type=str,
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
model = CASENet_resnet101(pretrained=True, num_classes=num_cls)
utils.load_pretrained_model(model, args.model)
crop_size = 512
mean_value = (104.008, 116.669, 122.675) #BGR
cls_names = get_sbd_class_names()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

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

    img_base_name_noext = os.path.splitext(os.path.basename(test_lst[idx_img]))[0]
    
    cls_activations = ['score_cls_side5_crop']
    # vis side edge activation
    for feature in ['score_edge_side1', 'score_edge_side2_crop', 'score_edge_side3_crop']:
        side = normalized_feature_map(net.blobs[feature].data[0][0, :, :])[0:height, 0:width]
        im = (side*255).astype(np.uint8)
        cv.imwrite(
            os.path.join(args.output_dir, img_base_name_noext+'_'+args.type+'_'+feature+'.png'),
            im)
    # vis side class activation
    side_cls = normalized_feature_map(np.transpose(net.blobs['score_cls_side5_crop'].data[0], (1, 2, 0)))
    for idx_cls in xrange(num_cls):
        side_cls_i = side_cls[0:height, 0:width, idx_cls]
        im = (side_cls_i * 255).astype(np.uint8)
        cv.imwrite(
            os.path.join(args.output_dir, img_base_name_noext+'_'+args.type+'_'+feature+'_'+cls_names[idx_cls]+'.png'),
            im)

    print 'processed: '+test_lst[idx_img]
    sys.stdout.flush()

print 'Done!'
