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

        input_size = 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        crop_size: 352
         mean_value: 104.008
        mean_value: 116.669
            mean_value: 122.675

    train_augmentation = transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
    # train_augmentation = transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]))
    train_dataset = SthData(
        root_img_folder,
        transform = torchvision.transforms.Compose([
                        train_augmentation,
                        Stack(roll=(args.basemodel_name in ['BNInception','InceptionV3'])),
                        ToTorchFormatTensor(div=(args.basemodel_name not in ['BNInception','InceptionV3'])),
                        normalize,
                        ]),
        batch_size=args.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    crop_size = input_size
    scale_size = input_size * 256 // 224
    val_dataset = SthData(
        val_jsonfile,
        root_img_folder,
        args.num_segments,
        transform = torchvision.transforms.Compose([
                        GroupScale(int(scale_size)),
                        GroupCenterCrop(crop_size),
                        Stack(roll=(args.basemodel_name in ['BNInception','InceptionV3'])),
                        ToTorchFormatTensor(div=(args.basemodel_name not in ['BNInception','InceptionV3'])),
                        normalize,
        ]),
        target_transform=None,
        mode="val",
        batch_size=args.batch_size/2)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size/2, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    # /2 because sometimes the same batch made validation out of memory.
    
    return train_loader, val_loader

