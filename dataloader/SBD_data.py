import torch
import torch.utils.data as data

import os
import numpy as np
import string
from PIL import Image

class SBDData(data.Dataset):
    
    def __init__(self, img_folder, label_folder, anno_txt, cls_num, img_transform):

        self.img_folder = img_folder
        self.label_folder = label_folder
        self.cls_num = cls_num
        self.img_transform = img_transform

        # Convert txt file to dict so that can use index to get filename.
        cnt = 0
        idx2name_dict = {}
        self.ids = []
        f = open(anno_txt, 'r')
        lines = f.readlines()
        for line in lines:
            row_data = line.split()
            img_name = row_data[0]
            label_name = row_data[1]
            idx2name_dict[cnt] = {}
            idx2name_dict[cnt]['img'] = img_name
            idx2name_dict[cnt]['label'] = label_name
            self.ids.append(cnt)
            cnt += 1

    def convert_num_to_bitfield(self, label_data, h, w):
        label_list = list(label_data)
        all_bit_tensor_list = []
        for n in label_list:
            # Convert a value to binary format in each bit.
            bitfield = np.asarray([int(digit) for digit in bin(n)[2:]])
            bit_tensor = torch.from_numpy(bitfield)
            actual_len = bit_tensor.size()[0]
            padded_bit_tensor = torch.cat((torch.zeros(self.cls_num-actual_len), bit_tensor), dim=1)
            all_bit_tensor_list.append(padded_bit_tensor)
        all_bit_tensor_list = torch.stack(all_bit_tensor_list).view(h, w, self.cls_num)
        
        return all_bit_tensor_list

    def __getitem__(self, index):
        img_name = self.idx2name_dict[index]['img']
        label_name = self.idx2name_dict[index]['label']
        img_path = os.path.join(self.img_folder, img_name)
        label_path = os.path.join(self.label_folder, label_name)

        # Load img into tensor
        img = Image.open(img_path).convert('RGB')
        processed_img = self.img_transform(img)
        h, w = img.size()

        # Load label into tensor
        label_data = np.fromfile(label_path, dtype=np.uint32)
        label_tensor = self.convert_num_to_bitfield(label_data)

        return processed_img, label_tensor

    def __len__(self):
        return len(self.ids)

