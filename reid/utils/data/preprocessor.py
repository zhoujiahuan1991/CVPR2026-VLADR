from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image

class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, strong_transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.strong_transform = strong_transform 
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        #print(self.dataset[0])
        try:
            fname, pid, camid, domain = self.dataset[index]
        except:
            fname, pid, camid, domain, _ = self.dataset[index]

        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')
        img_transformed = self.transform(img)

        if self.strong_transform is not None and self.transform is not None:
            strong_augemented = self.strong_transform(img)
            return img_transformed, strong_augemented, fpath, pid, camid, strong_augemented
        
        return img_transformed, fpath, pid, camid, domain

