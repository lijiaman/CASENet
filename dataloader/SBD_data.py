import torch
import torch.utils.data as data

import os
import numpy as np
import string
from PIL import Image
import time
import zipfile
import shutil

class SBDData(data.Dataset):
    
    def __init__(self, img_folder, label_folder, anno_txt, cls_num, input_size, img_transform):

        self.img_folder = img_folder
        self.label_folder = label_folder
        self.cls_num = cls_num
        self.input_size = input_size
        self.img_transform = img_transform

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

    def __getitem__(self, index):
        img_name = self.idx2name_dict[index]['img']
        label_name = self.idx2name_dict[index]['label']
        img_path = os.path.join(self.img_folder, img_name)
        label_path = os.path.join(self.label_folder, label_name)

        # Load img into tensor
        img = Image.open(img_path).convert('RGB')
        h, w = img.size
        processed_img = self.img_transform(img)

        # Load label into tensor
        # Read zip file and extract to npy, then load npy to numpy, delete numpy finally.
        zip_file = zipfile.ZipFile(label_path, 'r')
        tmp_folder = os.path.join("/ais/gobi4/fashion/edge_detection/tmp_npy", label_name.split('/')[-1])
        extract_data = zip_file.extract("label", tmp_folder)
        label_tensor = torch.from_numpy(np.resize(np.load(os.path.join(tmp_folder, "label")), (self.input_size, self.input_size, 3)))
        shutil.rmtree(tmp_folder) 
        return processed_img, label_tensor
        # processed_img: 3 X 352 X 352
        # label tensor: 352 X 352 X 3

    def __len__(self):
        return len(self.ids)

