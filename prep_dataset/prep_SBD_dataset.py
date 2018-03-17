import os
import numpy as np
import time

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
    train_anno_txt = "/ais/gobi4/fashion/edge_detection/data_aug/list_test.txt"
    #train_anno_txt = "/ais/gobi4/fashion/edge_detection/data_aug/list_train_aug.txt"
    val_anno_txt = "/ais/gobi4/fashion/edge_detection/data_aug/list_test.txt"

    input_size = 352
    normalize = transforms.Normalize(mean=[104.008, 116.669, 122.675], std=[1])

    train_dataset = SBDData(
        root_img_folder,
        root_label_folder,
        train_anno_txt,
        cls_num=args.cls_num,
        input_size=input_size,
        img_transform = transforms.Compose([
                        transforms.Resize([input_size, input_size]),
                        transforms.ToTensor(),
                        normalize,
                        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_dataset = SBDData(
        root_img_folder,
        root_label_folder,
        val_anno_txt,
        cls_num=args.cls_num,
        input_size=input_size,
        img_transform = transforms.Compose([
                        transforms.Resize([input_size, input_size]),
                        transforms.ToTensor(),
                        normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    return train_loader, val_loader

if __name__ == "__main__":
    args = config.get_args()
    train_loader, val_loader = get_dataloader(args)
    for i, (img, target) in enumerate(val_loader):
        print("target.size():{0}".format(target.size()))
        print("target:{0}".format(target))

