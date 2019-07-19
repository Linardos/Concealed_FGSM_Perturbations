# Implementation of the attack. Note that the original code was borrowed from PyTorch tutorials and build upon for the purposes of this project.

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models, utils
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import datasets
import datetime
from salient_bluring import miniatures
from salient_bluring.saliency_map_generation import infer_smap, SalBCE
from NIMA.model import NIMA, emd_loss
from PIL import Image

torch.manual_seed(10) # original test: 20


ROOT_PATH = "/home/linardos/Documents/pPrivacy"
PATH_TO_DATA = "../data/Places365/val_large"
PATH_TO_LABELS = "../data/Places365/places365_val.txt"

list_IDs = [line.rstrip('\n') for line in open("../data/Places365/MEPP19test.csv")]
BATCH_SIZE = 1
TILTSHIFT = True
USE_MAP = True

######################################################################
# Utility functions
# --------------
def minmax_normalization(X):
    return (X-X.min())/(X.max()-X.min())

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def load_weights(pt_model, device='cpu'):
    # Load stored model:
    temp = torch.load(pt_model, map_location=device)['state_dict']
    # Because of dataparallel there is contradiction in the name of the keys so we need to remove part of the string in the keys:.
    from collections import OrderedDict
    checkpoint = OrderedDict()
    for key in temp.keys():
        new_key = key.replace("module.","")
        checkpoint[new_key]=temp[key]

    return checkpoint


# ====================

# modules = list(model.children())[:-1] # The last layer is not compatible.
# print(checkpoint['state_dict'].keys())
# print(model.state_dict().keys())


######################################################################
# Implementation
# --------------
#
# Inputs
# ~~~~~~
#
# -  **epsilons** - List of epsilon values to use for the run. It is
#    important to keep 0 in the list because it represents the model
#    performance on the original test set. Also, intuitively we would
#    expect the larger the epsilon, the more noticeable the perturbations
#    but the more effective the attack in terms of degrading model
#    accuracy. Since the data range here is `[0,1]`, no epsilon
#    value should exceed 1.
#


# epsilons = [.01] #, .2, .25, .3]
# epsilons = [.15] #, .2, .25, .3]
pretrained_model = "./models/resnet50_places365.pth.tar"
use_cuda = True


######################################################################
# Model Under Attack
# ~~~~~~~~~~~~~~~~~~
#
# ==================== Load model ==================== #
# As mentioned, the model under attack is the ResNet model trained to recognize places.
# The purpose of this section is to define the model and dataloader,
# then initialize the model and load the pretrained weights.
#

#  Test dataset and dataloader declaration
transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
dataset = datasets.Places365(path_to_data=PATH_TO_DATA, path_to_labels=PATH_TO_LABELS, list_IDs=list_IDs,transform=transform_pipeline)
test_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE,
            shuffle=True)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network and load the weights
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(2048, 365) # To be made compatible to 365 number of classes instead of the original 1000
model.to(device)
checkpoint = load_weights(pretrained_model, device = device)
model.load_state_dict(checkpoint)

# Set the model in evaluation mode. There is no training a model when training for a defence, instead you optimize the input.
model.eval()




######################################################################
# FGSM Attack
# ~~~~~~~~~~~
#
# Now, we can define the function that creates the adversarial examples by
# perturbing the original inputs. The ``fgsm_attack`` function takes three
# inputs, *image* is the original clean image (`x`), *epsilon* is
# the pixel-wise perturbation amount (`\epsilon`), and *data_grad*
# is gradient of the loss w.r.t the input image
# (`\nabla_{x} J(\mathbf{\theta}, \mathbf{x}, y)`). The function
# then creates perturbed image as
#
# .. math:: perturbed\_image = image + epsilon*sign(data\_grad) = x + \epsilon * sign(\nabla_{x} J(\mathbf{\theta}, \mathbf{x}, y))
#
# Finally, in order to maintain the original range of the data, the
# perturbed image is clipped to range `[0,1]`.
#

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad, device, places_id, path_to_output):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    if USE_MAP:
        rmap = get_reverse_saliency(image) #Multiply by the reverse saliency map to mitigate perturbation on salient parts.
        rmap_noflat = rmap
        rmap_path = os.path.join("./adv_example", "rmap.png")
        ############
        # Using a sobel filter to find edges then blurring to spread the edginess. This gives us another map that allows us to avoid perturbing flat areas.
        from scipy import ndimage
        dx = ndimage.sobel(image.detach().cpu().squeeze(0).numpy(), 0)  # horizontal derivative
        dy = ndimage.sobel(image.detach().cpu().squeeze(0).numpy(), 1)  # vertical derivative
        mag = np.hypot(dx, dy)  # magnitude
        mag *= 255.0 / np.max(mag)  # normalize (Q&D)
        stdv = 10 # magnitude of blurring
        edginess = ndimage.filters.gaussian_filter(mag, stdv)
        edginess = np.transpose(edginess, (1,2,0))
        edginess = rgb2gray(edginess)
        edginess = torch.from_numpy(edginess).unsqueeze(0).unsqueeze(0)
        utils.save_image(minmax_normalization(edginess), os.path.join("./adv_example", "edginess.png"))
        edginess = minmax_normalization(edginess) #bring to 0-1 scale
        edginess = edginess.type(torch.FloatTensor).to(device)
        # rmap_noflat[edginess<edginess.mean()+abs(edginess.mean()/2)]=0 # Black out flat surfaces
        rmap_noflat[edginess<edginess.mean()]=0 # Black out flat surfaces
        rmap_noflat = minmax_normalization(rmap_noflat)

        perturbed_image = image + epsilon*sign_data_grad*rmap_noflat
        utils.save_image(minmax_normalization(rmap), rmap_path)
    else:
        rmap = None
        perturbed_image = image + epsilon*sign_data_grad

    perturbed_path = os.path.join(path_to_output, "Insight-DCU_e{}tshift_{}".format(epsilon, places_id))
    utils.save_image(minmax_normalization(perturbed_image), perturbed_path)

    # utils.save_image(minmax_normalization(image), os.path.join("./adv_example", "original.png"))
    # perturbed_image = torch.clamp(perturbed_image, -2, 2) # Changed to 95% confidence interval
    # Return the perturbed image
    if TILTSHIFT:
        perturbed_image = tiltshift(perturbed_path, rmap_path)
        perturbed_image = perturbed_image.convert('RGB') # important to add, By default PIL is agnostic about color spaces: https://stackoverflow.com/questions/50622180/does-pil-image-convertrgb-converts-images-to-srgb-or-adobergb/50623824
        perturbed_image = transform_pipeline(perturbed_image)
        perturbed_image = perturbed_image.unsqueeze(0)
        perturbed_image = perturbed_image.to(device)
        utils.save_image(minmax_normalization(perturbed_image), os.path.join("./adv_example", "blurred.png"))
    # to_pil = transforms.ToPILImage()
    # to_tensor = transforms.ToTensor()
    # perturbed_image = to_tensor(tiltshift(to_pil(perturbed_image.squeeze()))).unsqueeze()

    return perturbed_image, rmap
######################################################################
# Saliency & Blur
# ~~~~~~~~~~~
#

def get_reverse_saliency(img):
    _, reverse_map = infer_smap.map(img=img, weights="./salient_bluring/saliency_map_generation/salgan_salicon.pt", model=SalBCE.SalGAN(), device=device)
    return reverse_map

def tiltshift(img_path, rmap_path):
    img = Image.open(img_path)
    img = img.convert("RGB")
    rmap = Image.open(rmap_path)
    rmap = rmap.convert("RGB")
    # print(rmap.size)
    # print(img.size)
    x = miniatures.createMiniature(img, [], custom_mask=rmap)
    return x


######################################################################
# Testing Function
# ~~~~~~~~~~~~~~~~
#
# Finally, the central result of this tutorial comes from the ``test``
# function. Each call to this test function performs a full test step on
# the test set and reports a final accuracy. However, notice that
# this function also takes an *epsilon* input. This is because the
# ``test`` function reports the accuracy of a model that is under attack
# from an adversary with strength `epsilon`. More specifically, for
# each sample in the test set, the function computes the gradient of the
# loss w.r.t the input data (`data_grad`), creates a perturbed
# image with ``fgsm_attack`` (`perturbed_data`), then checks to see
# if the perturbed example is adversarial. In addition to testing the
# accuracy of the model, the function also saves and returns some
# successful adversarial examples to be visualized later.
#


def infer( model, device, test_loader, epsilon, path_to_output):

    start = datetime.datetime.now().replace(microsecond=0)

    # Accuracy counter
    correct = 0
    wrong_counter = 0
    adv_examples, original_aesthetics, final_aesthetics = [], [], []

    print("Initiating test with epsilon {} and tiltshift set to {}".format(epsilon, TILTSHIFT))
    # Loop over all examples in test set
    for i, (data, target, ID) in enumerate(test_loader):
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = F.softmax(output, dim=1)
        init_pred = init_pred.max(1, keepdim=True)[1] # get the index of the max log-probability

        #print(target)
        #print(init_pred)
        # exit()
        # print(init_pred)
        # exit()
        # print(F.log_softmax(output, dim=1))
        # target = torch.LongTensor([2]).to('cuda')
        # If the initial prediction is wrong, dont bother attacking, just move on
        # if init_pred.item() != target.item():
            # continue
        # print(F.log_softmax(output, dim=1).exp()) #Gives one hot encoding


        # Calculate the loss
        loss = F.nll_loss(output, target)
        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data, rmap = fgsm_attack(data, epsilon, data_grad, device, ID[0], path_to_output)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = F.softmax(output, dim=1).max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1

    end = datetime.datetime.now().replace(microsecond=0)
    # print("Wrong percentage: {}".format(wrong_counter/len(test_loader)))
    # Return the accuracy and an adversarial example
    final_acc = correct/len(test_loader)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}\t Time elapsed: {} \n".format(epsilon, correct, len(test_loader), final_acc, end-start))
    logfile = open("logfile.txt","a")
    logfile.write("\nInference started at {} and finished at {}".format(start, end))
    logfile.write("Epsilon: {}\tTest Accuracy = {} / {} = {}\t Time elapsed: {} \n".format(epsilon, correct, len(test_loader), final_acc, end-start))
    logfile.close()
    end = datetime.datetime.now().replace(microsecond=0)

    return end-start

if __name__ == '__main__':
    epsilons = [0.01, 0.05]
    for epsilon in epsilons:
        path_to_output = "../data/submissions/Insight-DCU_e{}tshift".format(epsilon)
        if not os.path.exists(path_to_output):
            os.mkdir(path_to_output)
        # Run test for each epsilon
        time = infer(model, device, test_loader, epsilon, path_to_output)
        print("Inference done after {}".format(time))

