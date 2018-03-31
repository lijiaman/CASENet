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

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from modules.CASENet import CASENet_resnet101
from prep_dataset.prep_SBD_dataset import RGB2BGR
from prep_dataset.prep_SBD_dataset import ToTorchFormatTensor

import utils.utils as utils

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
cls_names = get_sbd_class_names()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

input_size = 352
# normalize = transforms.Normalize(mean=[122.675, 116.669, 104.008], std=[1, 1, 1])
normalize = transforms.Normalize(mean=[104.008, 116.669, 122.675], std=[1, 1, 1])

img_transform = transforms.Compose([
                transforms.Resize([input_size, input_size]),
                RGB2BGR(roll=True),
                ToTorchFormatTensor(div=False),
                normalize,
                ])
#label_transform = transforms.Compose([
#                transforms.ToPILImage(),
#                transforms.Resize([input_size, input_size], interpolation=PIL.Image.NEAREST),
#                transforms.ToTensor(),
#                ])
#
#label_name = "label/test/2008_000003.zip"
#label_path = "/ais/gobi4/fashion/edge_detection/data_aug/label/test/2008_000003.zip"
#zip_file = zipfile.ZipFile(label_path, 'r')
#tmp_folder = os.path.join("/ais/gobi4/fashion/edge_detection/tmp_npy", label_name.split('/')[-2], label_name.split('/')[-1])
#extract_data = zip_file.extract("label", tmp_folder)
#np_data = np.load(os.path.join(tmp_folder, "label")) # H X W X NUM_CLASSES
#
#label_data = []
#for k in xrange(np_data.shape[2]):
#    label_tensor = label_transform(torch.from_numpy(np_data[:, :, k]).unsqueeze(0).float())
#    label_data.append(label_tensor.squeeze(0).long())
#label_data = torch.stack(label_data).transpose(0,1).transpose(1,2) # N X H X W -> H X W X N
#shutil.rmtree(tmp_folder)

for idx_img in xrange(len(test_lst)):
    img = Image.open(test_lst[idx_img]).convert('RGB')
    processed_img = img_transform(img).unsqueeze(0)
    processed_img = utils.check_gpu(0, processed_img)    
    score_feats1, score_feats2, score_feats3, score_feats5, score_fuse_feats = model(processed_img, for_vis=True)

    img_base_name_noext = os.path.splitext(os.path.basename(test_lst[idx_img]))[0]
   
    score_feats_list = [score_feats1, score_feats2, score_feats3]
    score_feats_str_list = ['feats1', 'feats2', 'feats3'] 

    # vis side edge activation
    for i in xrange(len(score_feats_list)):
        feature = score_feats_list[i]
        feature_str = score_feats_str_list[i]

        side = normalized_feature_map(feature.data[0][0, :, :].cpu().numpy())
        im = (side*255).astype(np.uint8)
        cv2.imwrite(
            os.path.join(args.output_dir, img_base_name_noext+'_'+feature_str+'.png'),
            im)
    # vis side class activation
    side_cls = normalized_feature_map(np.transpose(score_feats5.data[0].cpu().numpy(), (1, 2, 0)))
    for idx_cls in xrange(num_cls):
        side_cls_i = side_cls[:, :, idx_cls]
        im = (side_cls_i * 255).astype(np.uint8)
        cv2.imwrite(
            os.path.join(args.output_dir, img_base_name_noext+'_'+'feats5'+'_'+cls_names[num_cls-idx_cls-1]+'.png'),
            im)

    # vis class
    score_output = F.sigmoid(score_fuse_feats.transpose(1,3).transpose(1,2)).data[0].cpu().numpy()
    print("score_output.shape:{0}".format(score_output.shape))
    for idx_cls in xrange(num_cls):
        r = np.zeros((score_output.shape[0], score_output.shape[1]))
        g = np.zeros((score_output.shape[0], score_output.shape[1]))
        b = np.zeros((score_output.shape[0], score_output.shape[1]))
        rgb = np.zeros((score_output.shape[0], score_output.shape[1], 3))
        score_pred = score_output[:, :, idx_cls]
        score_pred_flag = (score_pred>0.5)
        r[score_pred_flag==1] = 255
        g[score_pred_flag==1] = 51
        b[score_pred_flag==1] = 51
        r[score_pred_flag==0] = 255
        g[score_pred_flag==0] = 255
        b[score_pred_flag==0] = 255
        rgb[:,:,0] = (r/255.0)
        rgb[:,:,1] = (g/255.0)
        rgb[:,:,2] = (b/255.0)
        plt.imsave(os.path.join(args.output_dir, img_base_name_noext+'_fused_pred_'+cls_names[num_cls-idx_cls-1]+'.png'), rgb) 
   
   # gt_data = label_data.numpy() 
   # for idx_cls in xrange(num_cls):
   #     r = np.zeros((gt_data.shape[0], gt_data.shape[1]))
   #     g = np.zeros((gt_data.shape[0], gt_data.shape[1]))
   #     b = np.zeros((gt_data.shape[0], gt_data.shape[1]))
   #     rgb = np.zeros((gt_data.shape[0], gt_data.shape[1], 3))
   #     score_pred_flag = gt_data[:, :, idx_cls]
   #     r[score_pred_flag==1] = 255
   #     g[score_pred_flag==1] = 51
   #     b[score_pred_flag==1] = 51
   #     r[score_pred_flag==0] = 255
   #     g[score_pred_flag==0] = 255
   #     b[score_pred_flag==0] = 255
   #     rgb[:,:,0] = (r/255.0)
   #     rgb[:,:,1] = (g/255.0)
   #     rgb[:,:,2] = (b/255.0)
   #     plt.imsave(os.path.join(args.output_dir, img_base_name_noext+'_gt_'+cls_names[num_cls-idx_cls-1]+'.png'), rgb) 

    print 'processed: '+test_lst[idx_img]
    sys.stdout.flush()

print 'Done!'
