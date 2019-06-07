import os
import numpy as np
import cv2
import torch
from torch.utils import data
from PIL import Image


class Places365(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, path_to_data, path_to_labels, list_IDs, resolution=None, transform=None):
        'Initialization'
        self.path_to_data = path_to_data
        self.list_IDs = list_IDs
        self.resolution = resolution
        self.path_to_labels = path_to_labels
        self.transform = transform
        self.d = {} #dictionary to match label to image
        with open(path_to_labels) as f:
            for line in f:
                (key, val) = line.split()
                self.d[str(key)] = int(val)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # Load image
        X = Image.open(os.path.join(self.path_to_data, ID))
        # print(X)
        X = X.convert('RGB')
        # print(X)
        if self.resolution!=None:
             X = X.resize(self.resolution)
        # deep learning networks traditionally share many parameters - if you didn't scale your inputs in a way that resulted in similarly-ranged feature values (ie: over the whole dataset by subtracting mean) sharing wouldn't happen very easily because to one part of the image weight w is a lot and to another it's too small.
        # print(X)
        if self.transform:
            X = self.transform(X)
        # print(X)
        # exit()
        # Match to label
        y = torch.tensor(self.d[ID])
        return X, y
