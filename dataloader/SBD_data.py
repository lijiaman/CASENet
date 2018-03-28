import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import os
import numpy as np
import string
import PIL
from PIL import Image
import time
import zipfile
import shutil
import pdb 
import h5py

class SBDData(data.Dataset):
    
    def __init__(self, img_folder, label_folder, anno_txt, hdf5_file_name, input_size, cls_num, img_transform, label_transform):

        self.img_folder = img_folder
        self.label_folder = label_folder
        self.input_size = input_size
        self.cls_num = cls_num
        self.img_transform = img_transform
        self.label_transform = label_transform

        self.h5_f = h5py.File(hdf5_file_name, 'r')

        # Convert txt file to dict so that can use index to get filename.
        cnt = 0
        self.idx2name_dict = {}
        self.ids = []
        f = open(anno_txt, 'r')
        lines = f.readlines()
        for line in lines:
            row_data = line.split()
            img_name = row_data[0]
            label_name = row_data[1]
            self.idx2name_dict[cnt] = {}
            self.idx2name_dict[cnt]['img'] = img_name
            self.idx2name_dict[cnt]['label'] = label_name
            self.ids.append(cnt)
            cnt += 1
            # break; # For temporal testing fit 1 sample

    def __getitem__(self, index):
        img_name = self.idx2name_dict[index]['img']
        label_name = self.idx2name_dict[index]['label']
        img_path = os.path.join(self.img_folder, img_name)

        # Load img into tensor
        img = Image.open(img_path).convert('RGB') # W X H
        processed_img = self.img_transform(img) # 3 X H X W

        np_data = self.h5_f['data/'+label_name.replace('/', '_').replace('bin', 'npy')]

        label_data = []
        for k in xrange(np_data.shape[2]):
            if np_data[:,:,k].sum() > 0:
                label_tensor = self.label_transform(torch.from_numpy(np_data[:, :, k]).unsqueeze(0).float())
            else: # ALL zeros, don't need transform
                label_tensor = torch.zeros(1, self.input_size, self.input_size).float()
            label_data.append(label_tensor.squeeze(0).long())
        label_data = torch.stack(label_data).transpose(0,1).transpose(1,2) # N X H X W -> H X W X N
        
        return processed_img, label_data
        # processed_img: 3 X 352(H) X 352(W)
        # label tensor: 352(H) X 352(W) X 20

    def __len__(self):
        return len(self.ids)

