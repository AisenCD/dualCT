
from __future__ import print_function, division

# https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
from random import random, randint

# Ignore warnings
import warnings
import pdb

warnings.filterwarnings("ignore")


# mode='train'
# root = r'D:\workspace\data\IVDM3Seg\Training'
# root = r'D:\workspace\data\dual_ct\patients_20220206\Training\img_all'

def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        train_fat_path = os.path.join(root, 'Z_Effective')
        train_inn_path = os.path.join(root, 'Iodine_Density')
        train_opp_path = os.path.join(root, 'Iodine_no_Water')
        train_wat_path = os.path.join(root, 'MonoE_70keV')
        train_mono40_path = os.path.join(root, 'MonoE_40keV')
        train_mono100_path = os.path.join(root, 'MonoE_100keV')
        train_mask_path = os.path.join(root, 'label')

        images_fat = os.listdir(train_fat_path)
        images_inn = os.listdir(train_inn_path)
        images_opp = os.listdir(train_opp_path)
        images_wat = os.listdir(train_wat_path)
        images_mono40 = os.listdir(train_mono40_path)
        images_mono100 = os.listdir(train_mono100_path)        
        labels = os.listdir(train_mask_path)

        images_fat.sort()
        images_inn.sort()
        images_opp.sort()
        images_wat.sort()
        images_mono40.sort()
        images_mono100.sort()        
        labels.sort()

        for it_f,it_i,it_o,it_w,img_mono40,img_mono100,it_gt in zip(images_fat,images_inn,images_opp,images_wat,images_mono40,images_mono100,labels):
            item = (os.path.join(train_fat_path, it_f),
                    os.path.join(train_inn_path, it_i),
                    os.path.join(train_opp_path, it_o),
                    os.path.join(train_wat_path, it_w),
                    os.path.join(train_mono40_path, img_mono40),
                    os.path.join(train_mono100_path, img_mono100),
                    os.path.join(train_mask_path, it_gt))
            items.append(item)
            
    elif mode == 'val':
        train_fat_path = os.path.join(root, 'Z_Effective')
        train_inn_path = os.path.join(root, 'Iodine_Density')
        train_opp_path = os.path.join(root, 'Iodine_no_Water')
        train_wat_path = os.path.join(root, 'MonoE_70keV')
        train_mono40_path = os.path.join(root, 'MonoE_40keV')
        train_mono100_path = os.path.join(root, 'MonoE_100keV')        
        train_mask_path = os.path.join(root, 'label')

        images_fat = os.listdir(train_fat_path)
        images_inn = os.listdir(train_inn_path)
        images_opp = os.listdir(train_opp_path)
        images_wat = os.listdir(train_wat_path)
        images_mono40 = os.listdir(train_mono40_path)
        images_mono100 = os.listdir(train_mono100_path)        
        labels = os.listdir(train_mask_path)

        images_fat.sort()
        images_inn.sort()
        images_opp.sort()
        images_wat.sort()
        images_mono40.sort()
        images_mono100.sort()        
        labels.sort()

        for it_f,it_i,it_o,it_w,img_mono40,img_mono100,it_gt in zip(images_fat,images_inn,images_opp,images_wat,images_mono40,images_mono100,labels):
            item = (os.path.join(train_fat_path, it_f),
                    os.path.join(train_inn_path, it_i),
                    os.path.join(train_opp_path, it_o),
                    os.path.join(train_wat_path, it_w),
                    os.path.join(train_mono40_path, img_mono40),
                    os.path.join(train_mono100_path, img_mono100),
                    os.path.join(train_mask_path, it_gt))
            items.append(item)

    return items


class MedicalImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode, root_dir, transform=None, mask_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.imgs = make_dataset(root_dir, mode)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        fat_path, inn_path,opp_path,wat_path,mono40_path,mono100_path,mask_path = self.imgs[index]

        img_f = Image.open(fat_path)#.convert('L')
        img_i = Image.open(inn_path)#.convert('L')
        img_o = Image.open(opp_path)#.convert('L')
        img_w = Image.open(wat_path)#.convert('L')
        img_mono40 = Image.open(mono40_path)#.convert('L')
        img_mono100 = Image.open(mono100_path)#.convert('L')        
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            img_f = self.transform(img_f)
            img_i = self.transform(img_i)
            img_o = self.transform(img_o)
            img_w = self.transform(img_w)
            img_mono40 = self.transform(img_mono40)
            img_mono100 = self.transform(img_mono100)            
            mask = self.mask_transform(mask)

        return [img_f,img_i,img_o,img_w,img_mono40,img_mono100, mask, fat_path]
