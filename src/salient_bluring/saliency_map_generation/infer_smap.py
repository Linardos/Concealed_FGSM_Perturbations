import cv2
import os
import sys
import numpy as np
import pickle
import torch
from torchvision import transforms, utils
from torch.autograd import Variable

def map(img, weights, model):
    frame_size = (192, 256)
    dtype = torch.FloatTensor

    img = cv2.imread(img)
    #img = cv2.resize(img, (frame_size[1], frame_size[0]), interpolation=cv2.INTER_AREA)
    img = torch.Tensor(img)
    img = Variable(img.type(dtype).transpose(0,1), requires_grad=False)
    img = img.unsqueeze(0).transpose(1,3)
    print(img.size())

    model.salgan.load_state_dict(torch.load(weights, map_location='cpu')['state_dict'])
    saliency_map = model.forward(input_ = img)
    saliency_map = saliency_map

    post_process_saliency_map = (saliency_map-torch.min(saliency_map))/(torch.max(saliency_map)-torch.min(saliency_map))
    post_process_saliency_map = torch.nn.functional.interpolate(post_process_saliency_map, (img.size()[2], img.size()[3]))
    utils.save_image(post_process_saliency_map, "saliency_map.png")

    reverse_smap = torch.ones(post_process_saliency_map.size())
    reverse_smap -= post_process_saliency_map
    print(reverse_smap.size())
    utils.save_image(reverse_smap, "reverse_saliency_map.png")

    return(post_process_saliency_map, reverse_smap)

if __name__ == '__main__':
    import SalBCE
    map(sys.argv[1], "salgan_salicon.pt", SalBCE.SalGAN())
