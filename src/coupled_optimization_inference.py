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
import copy
import datasets
import datetime
from NIMA.model import NIMA, emd_loss
from PIL import Image
from collections import OrderedDict

torch.manual_seed(10)

ROOT_PATH = "/home/linardos/Documents/pPrivacy"
PATH_TO_DATA = "../data/Places365/val_large"
PATH_TO_LABELS = "../data/Places365/places365_val.txt"
list_IDs = [line.rstrip('\n') for line in open("../data/Places365/MEPP19test.csv")]
BATCH_SIZE = 1
COUPLED = True
######################################################################
# Utility functions
# --------------
def minmax_normalization(X):
    return (X-X.min())/(X.max()-X.min())

def standardization(X):
    return X.sub_(X.mean()).div_(X.std())

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


pretrained_model = "./models/resnet50_places365.pth.tar"
use_cuda = True


######################################################################
# Model Under Attack
# ~~~~~~~~~~~~~~~~~~
#
# The model under attack is the ResNet model trained to recognize places.
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
attack_model = models.resnet50(pretrained=False)
attack_model.fc = nn.Linear(2048, 365) # To be made compatible to 365 number of classes instead of the original 1000
attack_model.to(device)
checkpoint = load_weights(pretrained_model, device = device)
attack_model.load_state_dict(checkpoint)

# Set the attack_model in evaluation mode. There is no training of attack_model when setting up a defence, instead you optimize the input.
attack_model.eval()


# Aesthetics Model
# ~~~~~~~~~~~~~~~~~~

# for child in aesthetics_model.children():
#     print(child)
base_model = models.vgg16(pretrained=False)
# old_model = NIMA(base_model)
# old_model.load_state_dict(torch.load("./NIMA/epoch-12.pkl"))

aesthetics_model = NIMA(base_model)
aesthetics_model.load_state_dict(torch.load("./NIMA/epoch-12.pkl"))

aesthetics_model.to(device)
aesthetics_model.eval()

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
    # print(type(image))
    perturbed_image = image + epsilon*sign_data_grad

    utils.save_image(minmax_normalization(perturbed_image), os.path.join(path_to_output, "Insight-DCU_e{}copt_{}".format(epsilon, places_id)))
    # perturbed_image = torch.clamp(perturbed_image, -2, 2) # Changed to 95% confidence interval
    # Return the perturbed image

    # to_pil = transforms.ToPILImage()
    # to_tensor = transforms.ToTensor()
    # perturbed_image = to_tensor(tiltshift(to_pil(perturbed_image.squeeze()))).unsqueeze()

    return perturbed_image


######################################################################
# Testing Function
# ~~~~~~~~~~~~~~~~
#
# Finally, the central result of this tutorial comes from the ``test``
# function. Each call to this test function performs a full test step on
# the test set and reports a final accuracy. However, notice that
# this function also takes an *epsilon* input. This is because the
# ``test`` function reports the accuracy of a attack_model that is under attack
# from an adversary with strength `epsilon`. More specifically, for
# each sample in the test set, the function computes the gradient of the
# loss w.r.t the input data (`data_grad`), creates a perturbed
# image with ``fgsm_attack`` (`perturbed_data`), then checks to see
# if the perturbed example is adversarial. In addition to testing the
# accuracy of the attack_model, the function also saves and returns some
# successful adversarial examples to be visualized later.
#


def infer( attack_model, aesthetics_model, device, test_loader, epsilon, path_to_output):

    start = datetime.datetime.now().replace(microsecond=0)

    # Accuracy counter
    correct = 0
    wrong_counter = 0
    adv_examples = []
    print("Initiating test with epsilon {} and coupled set to {}".format(epsilon, COUPLED))

    # Set the highest goal
    aesthetics_goal = torch.zeros(10).unsqueeze(0)
    aesthetics_goal[-1] = 1 #Last index means score 10
    aesthetics_goal = aesthetics_goal.to(device)
    # Loop over all examples in test set
    for i, (data, target, ID) in enumerate(test_loader):

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        data_copy = data.clone()
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        data_copy.requires_grad = True
        # Forward pass the data through the attack_model
        attack_output = attack_model(data)
        init_pred = F.softmax(attack_output, dim=1)
        init_pred = init_pred.max(1, keepdim=True)[1] # get the index of the max log-probability

        # data_copy = data_copy.resize_
        # transforms.Resize(interpolation=2)
        aesthetics_output = aesthetics_model(data_copy)
        # print(aesthetics_output.size())
        # print(aesthetics_goal.size())
        # print(aesthetics_output)

        # print(F.log_softmax(attack_output, dim=1))
        # target = torch.LongTensor([2]).to('cuda')
        # If the initial prediction is wrong, dont bother attacking, just move on
        # print(F.log_softmax(attack_output, dim=1).exp()) #Gives one hot encoding


        # Calculate the loss
        attack_loss = F.nll_loss(attack_output, target)
        # Zero all existing gradients
        attack_model.zero_grad()
        # Calculate gradients of attack_model in backward pass
        attack_loss.backward()

        aesthetics_loss = emd_loss(aesthetics_output, aesthetics_goal)
        aesthetics_model.zero_grad()
        aesthetics_loss.backward()
        # Collect datagrad
        data_grad = data.grad.data
        data_aesth_grad = data_copy.grad.data

        # The gradient of two different networks will be on a completely different scale. We need to make them comparable. We only care for the sign in the end, so standardize to mean 0 and std 1
        if COUPLED:
            data_grad = standardization(data_grad)
            data_aesth_grad = standardization(data_aesth_grad)
            data_grad+=data_aesth_grad

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad, device, ID[0], path_to_output)

        # Re-classify the perturbed image
        attack_output = attack_model(perturbed_data)

        # Check for success
        final_pred = F.softmax(attack_output, dim=1).max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1



    # Calculate final accuracy for this epsilon
    # print("Wrong percentage: {}".format(wrong_counter/len(test_loader)))
    # Return the accuracy and an adversarial example
    end = datetime.datetime.now().replace(microsecond=0)
    final_acc = correct/len(test_loader)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}\t Time elapsed: {} \n".format(epsilon, correct, len(test_loader), final_acc, end-start))
    logfile = open("logfile.txt","a")
    logfile.write("\nInference started at {} and finished at {}".format(start, end))
    logfile.write("Epsilon: {}\nTest Accuracy = {} / {} = {}\t Time elapsed: {} \n".format(epsilon, correct, len(test_loader), final_acc, end-start))
    logfile.close()

    return end-start


if __name__ == '__main__':
    epsilons = [0.01, 0.05]
    for epsilon in epsilons:
        path_to_output = "../data/submissions/Insight-DCU_e{}copt".format(epsilon)
        if not os.path.exists(path_to_output):
            os.mkdir(path_to_output)
        # Run test for each epsilon
        time = infer(attack_model, aesthetics_model, device, test_loader, epsilon, path_to_output)
        print("Inference done after {}".format(time))


