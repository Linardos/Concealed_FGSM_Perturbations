import scipy
from scipy import ndimage
import numpy as np
import torch
from PIL import Image
from salient_bluring.saliency_map_generation import infer_smap, SalBCE


def get_reverse_saliency(img):
    _, reverse_map = infer_smap.map(img=img, weights="./salient_bluring/saliency_map_generation/salgan_salicon.pt", model=SalBCE.SalGAN())
    return reverse_map

def rgb2gray(rgb):
    # print(rgb[...,:3].shape)
    # exit()
    return np.dot(rgb, [0.2989, 0.5870, 0.1140]) # ... Ellipsis object, is same as using :,: it covers what is not defined.

im = scipy.misc.imread('../data/trial_images/bike.jpg')
im = im.astype('int32')
im = torch.from_numpy(im)
print(type(im))
dx = ndimage.sobel(im, 0)  # horizontal derivative
dy = ndimage.sobel(im, 1)  # vertical derivative
mag = np.hypot(dx, dy)  # magnitude
mag *= 255.0 / np.max(mag)  # normalize (Q&D)
scipy.misc.imsave('./sobel_trials/sobel.jpg', mag)

stdv = 10 # magnitude of blurring
final = ndimage.filters.gaussian_filter(mag, stdv)
print(final.shape)
final = rgb2gray(final)
# final = Image.fromarray(final, 'L')
print(final)
scipy.misc.imsave('./sobel_trials/sobelblurred{}.jpg'.format(stdv), final)

# rmap = get_reverse_saliency(im.permute(2,0,1).unsqueeze(0))
# scipy.misc.imsave('./sobel_trials/rmap.jpg', final)


