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
from NIMA.model import NIMA, NIMA2, emd_loss
from PIL import Image
from collections import OrderedDict

torch.manual_seed(20)

ROOT_PATH = "/home/linardos/Documents/pPrivacy"
PATH_TO_DATA = "../data/Places365/val_large"
PATH_TO_LABELS = "../data/Places365/places365_val.txt"
BATCH_SIZE = 1
TEST_NUMBER = 100 #

######################################################################
# Utility functions
# --------------
def minmax_normalization(X):
    return (X-X.min())/(X.max()-X.min())

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


epsilons = [0, .01, .025, .05] #, .2, .25, .3]
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
dataset = datasets.Places365(path_to_data=PATH_TO_DATA, path_to_labels=PATH_TO_LABELS, list_IDs = os.listdir(PATH_TO_DATA), transform=transform_pipeline)
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

aesthetics_model = NIMA2(base_model)
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
def fgsm_attack(image, epsilon, data_grad, device):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    # print(type(image))
    perturbed_image = image + epsilon*sign_data_grad

    utils.save_image(minmax_normalization(perturbed_image), os.path.join("./adv_example", "perturbed.png"))
    utils.save_image(minmax_normalization(image), os.path.join("./adv_example", "original.png"))
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


def test( attack_model, device, test_loader, epsilon ):

    start = datetime.datetime.now().replace(microsecond=0)

    # Accuracy counter
    correct = 0
    wrong_counter = 0
    adv_examples = []
    print("Initiating test with epsilon {}".format(epsilon))

    # Set the highest goal
    aesthetics_goal = torch.zeros(10).unsqueeze(0)
    aesthetics_goal[-1] = 1 #Last index means score 10
    aesthetics_goal = aesthetics_goal.to(device)
    # Loop over all examples in test set
    for i, (data, target) in enumerate(test_loader):

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        data_copy = data
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
        if i == TEST_NUMBER:
            break
        if init_pred.item() != target.item():
            continue
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
        data_copy_grad = data_copy.grad.data
        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad, device)

        # Re-classify the perturbed image
        attack_output = attack_model(perturbed_data)

        # Check for success
        final_pred = F.softmax(attack_output, dim=1).max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                original_ex = data.squeeze().detach().cpu()#.numpy()
                adv_ex = perturbed_data.squeeze().detach().cpu()#.numpy()
                adv_examples.append( (original_ex, adv_ex) )
        else:
            # Save some adv examples for visualization later
            # print("got some")
            if len(adv_examples) < 5:

                original_ex = data.squeeze().detach().cpu()#.numpy()
                adv_ex = perturbed_data.squeeze().detach().cpu()#.numpy()
                adv_examples.append( (original_ex, adv_ex) )


    # Calculate final accuracy for this epsilon
    final_acc = correct/float(TEST_NUMBER)
    # print("Wrong percentage: {}".format(wrong_counter/TEST_NUMBER))
    # Return the accuracy and an adversarial example
    end = datetime.datetime.now().replace(microsecond=0)

    print("Epsilon: {}\tTest Accuracy = {} / {} = {}\t Time elapsed: {}".format(epsilon, correct, TEST_NUMBER, final_acc, end-start)) #replace TEST_NUMBER with len(test_loader) when done with testing

    return final_acc, adv_examples


######################################################################
# Run Attack
# ~~~~~~~~~~
#
# The last part of the implementation is to actually run the attack. Here,
# we run a full test step for each epsilon value in the *epsilons* input.
# For each epsilon we also save the final accuracy and some successful
# adversarial examples to be plotted in the coming sections. Notice how
# the printed accuracies decrease as the epsilon value increases. Also,
# note the `\epsilon=0` case represents the original test accuracy,
# with no attack.
#

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(attack_model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)


######################################################################
# Results
# -------
#
# Accuracy vs Epsilon
# ~~~~~~~~~~~~~~~~~~~
#
# The first result is the accuracy versus epsilon plot. As alluded to
# earlier, as epsilon increases we expect the test accuracy to decrease.
# This is because larger epsilons mean we take a larger step in the
# direction that will maximize the loss. Notice the trend in the curve is
# not linear even though the epsilon values are linearly spaced. For
# example, the accuracy at `\epsilon=0.05` is only about 4% lower
# than `\epsilon=0`, but the accuracy at `\epsilon=0.2` is 25%
# lower than `\epsilon=0.15`. Also, notice the accuracy of the attack_model
# hits random accuracy for a 10-class classifier between
# `\epsilon=0.25` and `\epsilon=0.3`.
#

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(epsilons[0], epsilons[-1], step=.01))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.savefig("Epsilons.png")
# plt.show()


######################################################################
# Sample Adversarial Examples
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Remember the idea of no free lunch? In this case, as epsilon increases
# the test accuracy decreases **BUT** the perturbations become more easily
# perceptible. In reality, there is a tradeoff between accuracy
# degredation and perceptibility that an attacker must consider. Here, we
# show some examples of successful adversarial examples at each epsilon
# value. Each row of the plot shows a different epsilon value. The first
# row is the `\epsilon=0` examples which represent the original
# “clean” images with no perturbation. The title of each image shows the
# “original classification -> adversarial classification.” Notice, the
# perturbations start to become evident at `\epsilon=0.15` and are
# quite evident at `\epsilon=0.3`. However, in all cases humans are
# still capable of identifying the correct class despite the added noise.
#

# # Plot several examples of adversarial samples at each epsilon
# cnt = 0
# plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        orig, ex = examples[i][j]

        utils.save_image(minmax_normalization(orig), "./adv_example/original_e{}.png".format(epsilons[i]))
        utils.save_image(minmax_normalization(ex), "./adv_example/example_e{}.png".format(epsilons[i]))

#         cnt += 1
#         plt.subplot(len(epsilons),len(examples[0]),cnt)
#         plt.xticks([], [])
#         plt.yticks([], [])
#         if j == 0:
#             plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
#         plt.title("{} -> {}".format(orig, adv))
#         plt.imshow(ex, cmap="gray")
# plt.tight_layout()
# plt.savefig("QAnal.png")
# plt.show()
