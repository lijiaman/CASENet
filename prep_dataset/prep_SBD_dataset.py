import os
import numpy as np
import time

import PIL
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import sys
sys.path.append("../")

from dataloader.SBD_data import SBDData

import config

def get_dataloader(args):
    # Define data files path.
    root_img_folder = "/ais/gobi4/fashion/edge_detection/data_aug" 
    root_label_folder = "/ais/gobi4/fashion/edge_detection/data_aug"
    # train_anno_txt = "/ais/gobi4/fashion/edge_detection/data_aug/list_test.txt"
    train_anno_txt = "/ais/gobi4/fashion/edge_detection/data_aug/list_train.txt"
    val_anno_txt = "/ais/gobi4/fashion/edge_detection/data_aug/list_test.txt"
    #train_hdf5_file = "/ais/gobi6/jiaman/github/CASENet/utils/test_label_binary_np.h5"
    train_hdf5_file = "/ais/gobi6/jiaman/github/CASENet/utils/train_label_binary_np.h5"
    val_hdf5_file = "/ais/gobi6/jiaman/github/CASENet/utils/test_label_binary_np.h5"

    input_size = 352
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = SBDData(
        root_img_folder,
        root_label_folder,
        train_anno_txt,
        train_hdf5_file,
        input_size,
        cls_num=args.cls_num,
        img_transform = transforms.Compose([
                        transforms.Resize([input_size, input_size]),
                        transforms.ToTensor(),
                        normalize,
                        ]),
        label_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize([input_size, input_size], interpolation=PIL.Image.NEAREST),
                        transforms.ToTensor(),
                        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_dataset = SBDData(
        root_img_folder,
        root_label_folder,
        val_anno_txt,
        val_hdf5_file,
        input_size,
        cls_num=args.cls_num,
        img_transform = transforms.Compose([
                        transforms.Resize([input_size, input_size]),
                        transforms.ToTensor(),
                        normalize,
                        ]),
        label_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize([input_size, input_size], interpolation=PIL.Image.NEAREST),
                        transforms.ToTensor(),
                        ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    return train_loader, val_loader

if __name__ == "__main__":
    args = config.get_args()
    args.batch_size = 1
    train_loader, val_loader = get_dataloader(args)
    for i, (img, target) in enumerate(val_loader):
        print("target.size():{0}".format(target.size()))
        print("target:{0}".format(target))
        break;
