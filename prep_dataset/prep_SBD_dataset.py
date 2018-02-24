import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import sys
sys.path.append("../")

from dataloader.SBD_data import SBDData

def get_dataloader(args):
    # Define data files path.
    root_img_folder = "/ais/gobi4/fashion/edge_detection/data_aug/" 
    root_label_folder = "/ais/gobi4/fashion/edge_detection/data_aug/"
    train_anno_txt = "/ais/gobi4/fashion/edge_detection/data_aug/list_train_aug.txt"
    val_ano_txt = "/ais/gobi4/fashion/edge_detection/data_aug/list_test.txt"

    input_size = 352
    normalize = transforms.Normalize(mean=[104.008, 116.669, 122.675])

    train_dataset = SBDData(
        root_img_folder,
        root_label_folder,
        train_anno_txt,
        cls_num=args.cls_num,
        img_transform = torchvision.transforms.Compose([
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
        img_transform = torchvision.transforms.Compose([
                        normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size/2, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    # /2 because sometimes the same batch made validation out of memory.
    
    return train_loader, val_loader

