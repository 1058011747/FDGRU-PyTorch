import os
import torch.utils.data as data
import numpy as np
import torch
import json
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage import io,transform
import glob

class CWRUdata(data.Dataset):
    def __init__(self, DATA_DIR, split, transform=None, target_transform=None):
        super(CWRUdata,self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        files = os.listdir(DATA_DIR)
        files.sort()
        cate=[DATA_DIR+x for x in files if os.path.isdir(DATA_DIR+x)]
        imgs=[]
        labels=[]
        for idx,folder in enumerate(cate):
            for im in glob.glob(folder+'/*.png'):
                img=io.imread(im)
                img = img.reshape(32,32)
                img = np.expand_dims(img, axis=0)
                imgs.append(img)
                labels.append(idx)
        self.img = imgs
        self.label = labels

    def __getitem__(self, index):
        img = self.img[index]
        label = self.label[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        if self.split == 'train':
            return len(self.img)
        else:
            return len(self.img)