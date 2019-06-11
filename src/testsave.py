# Somehow messes the image, but how?

import os
import os
import numpy as np
import cv2
import torch
from torch.utils import data
from PIL import Image
from torchvision import utils, transforms


# Load image
X = Image.open("/home/linardos/Documents/pPrivacy/data/Places365/val_large/Places365_val_00000001.jpg")
# print(X)
# X = X.convert('RGB')

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
X = transform(X)

# Normalization messes it up!!! rescale to 0 - 1
X = (X-X.min())/(X.max()-X.min())

utils.save_image(X, os.path.join("./adv_example", "testsave.png"))

