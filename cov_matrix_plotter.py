import torch
import torchvision
import os,  sys
import os, argparse, time, subprocess, io, shlex, pickle, pprint
import numpy as np
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from vonenet import get_dn_model_test

batch_size = 128
model_arch = "resnet18"
normalization = "vonenet"
visual_degrees = 2
stride = 2
ksize = 25
k_exc = 25
simple_channels = 32
complex_channels = simple_channels
image_size = 64
sf_min = 0.5
sf_max = 11.2

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_built():
    device = "mps"
    mps = True
else:
    device = "cpu"

parser = argparse.ArgumentParser(description='Covariance plotting')

parser.add_argument('--cov_path', type = str,
                    help='path to folder that contains cov matrix and filters')
parser.add_argument('-o', '--output_path', default=None,
                    help='path for storing ')
parser.add_argument("--ckpt", type = str, default="", help = "Ckpt to load from.")
parser.add_argument("--trainable_vonenetdn",  action="store_true", help = "Whether to train divisive norm scalars.")
parser.add_argument("--plots_edge", type = int, default = 1)
parser.add_argument("--images_edge", type = int, default = 1)

parser.add_argument("--images",  action="store_true", help = "Plot images.")
parser.add_argument("--gabors",  action="store_true", help = "PLot Gabors.")
parser.add_argument("--rfs",  action="store_true", help = "Plot rfs.")
parser.add_argument("--denominators",  action="store_true", help = "Plot denominators.")
parser.add_argument("--normed",  action="store_true", help = "Plot norm outputs.")


FLAGS, FIRE_FLAGS = parser.parse_known_args()

print(FLAGS)

images = FLAGS.images
gabors = FLAGS.gabors
cov_matrix_rfs = FLAGS.rfs
denominators = FLAGS.denominators
normed = FLAGS.normed


plots_edge = FLAGS.plots_edge
plots_count = plots_edge ** 2

images_edge = FLAGS.images_edge
images_count = images_edge ** 2


output_path = FLAGS.cov_path
ckpt = FLAGS.ckpt

cov_matrix = None
filters_r = None
filters_c = None
use_checkpoint = False

if ckpt != "":
    checkpoint = torch.load(ckpt)
    print("Loaded from: ", ckpt)
    use_checkpoint = True

cov_matrix = torch.load(output_path+"/cov_matrix.pt").to(device)
filters_r = torch.load(output_path+"/real_filters.pt").to(device)
filters_c = torch.load(output_path+"/imaginary_filters.pt").to(device)

def data():
    dataset = torchvision.datasets.ImageFolder(
        os.path.join("tiny-imagenet-200", 'train'),
        torchvision.transforms.Compose([
            # torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomAffine(degrees=30, translate=(0.05, 0.05), scale=(1, 1.2)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ]))
    data_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=128,
                                                shuffle=False,
                                                num_workers=2,
                                                pin_memory=True)

    return data_loader

torch.manual_seed(0)
train_data = data()

for step, dt in enumerate(train_data):

    if step == 0:
        first_sample = dt
        break

if images:
    # Create a figure and axis object using Matplotlib
    cnt = images_count
    rows = int(np.sqrt(cnt))

    fig, axes = plt.subplots(rows, rows, figsize = (2*rows,2*rows))

    # Flatten the axes array to make it easier to iterate over
    axes = axes.flatten()

    # Loop through each subplot and plot the heatmap
    for i, ax in enumerate(axes):

        img = first_sample[0][i].permute(1,2,0)
        ax.imshow(img)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.savefig("plots/images/first_sample_images.png")

# Generate intermediate outputs

vondn = get_dn_model_test(map_location=device, pretrained = False, simple_channels=simple_channels, gabor_seed=0, complex_channels=complex_channels, model_arch="resnet18", noise_mode = None, k_exc=25, ksize=25, stride = stride, image_size=image_size, visual_degrees=visual_degrees,
                  filters_r = filters_r, filters_c = filters_c, cov_matrix = cov_matrix, trainable=FLAGS.trainable_vonenetdn)

def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

if use_checkpoint:
    print("Using checkpoint ", FLAGS.ckpt)
    ckpts = remove_data_parallel(checkpoint["state_dict"])
    vondn.load_state_dict(ckpts)

voneblockdn = vondn[0]

outputs_inter = voneblockdn.gabors_f(first_sample[0].to(device))

if gabors:
    sz = outputs_inter.shape[1]
    btc = 0 # index of img in batch

    for btc in tqdm.tqdm(range(images_count)):
        rows = int(np.sqrt(sz))
        fil = outputs_inter[btc][:sz,::]
        print(fil.shape)
        # Create a figure and axis object using Matplotlib
        fig, axes = plt.subplots(rows, rows, figsize=(2*rows, 2*rows))

        # Flatten the axes array to make it easier to iterate over
        axes = axes.flatten()

        print("Plotting gabor outputs.")
        # Loop through each subplot and plot the heatmap
        for i, ax in enumerate(axes):
            if i < rows * rows:
                ax.imshow(fil[i].to("cpu"))
            else:
                ax.axis('off')  # Turn off axis for empty subplots

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plot
        plt.savefig(f"plots/gabors/gabor_outputs_{btc}.png")

dimm = image_size // stride
rho = dimm // 2

inds = np.int64(np.random.multivariate_normal([rho,rho], [[rho,0],[0,rho]], 10))
shift = np.random.choice([-1,1])

u = np.arange(0,dimm**2).reshape(dimm,dimm)
positions = [u[pos[0]][pos[1]] for pos in inds]

if cov_matrix_rfs:
    for ind in tqdm.tqdm(positions):

        shift = np.random.choice(np.arange(-3,3)) * image_size//stride + np.random.choice(np.arange(-4*image_size//stride//7, +4*image_size//stride//7))

        covvies = cov_matrix[ind].reshape(simple_channels+complex_channels, image_size//stride, image_size//stride)

        sz = covvies.shape[0]

        rows = int(np.sqrt(sz))
        fil = covvies
        # Create a figure and axis object using Matplotlib
        fig, axes = plt.subplots(rows, rows, figsize=(2*rows, 2*rows))

        # Flatten the axes array to make it easier to iterate over
        axes = axes.flatten()

        print("Plotting receptive fields")

        # Loop through each subplot and plot the heatmap
        for i, ax in enumerate(axes):
            if i < rows * rows:
                sns.heatmap(fil[i].to("cpu"), ax=ax, cmap="viridis", cbar=False)
            else:
                ax.axis('off')  # Turn off axis for empty subplots

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plot
        plt.savefig(f"plots/cov_matrix_rfs/{ind}.png")


if denominators:
    if ckpt:

        res = voneblockdn.dn.denominator(outputs_inter)

    else:

        beta = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)
        cc = simple_channels+complex_channels
        cxc = torch.tensor(torch.rand(cc, cc)) / 0.5 / 1e4
        norm_mults = nn.Parameter(cxc, requires_grad=True).to(device)

        inter = outputs_inter.permute(0,3,2,1)
        result = torch.einsum('bxyc,cd->bxyc', inter, norm_mults)
        result = result.permute(0, 3, 1, 2)

        trial = result.reshape(-1, np.prod(list(result.shape[1:])))
        div = cov_matrix@trial.T
        res = div.T.reshape(outputs_inter.shape)

    sz = res.shape[1]


    for btc in tqdm.tqdm(range(plots_count)):

        rows = int(np.sqrt(sz))
        fil = res[btc][:sz,::]

        # Create a figure and axis object using Matplotlib
        fig, axes = plt.subplots(rows, rows, figsize=(4*rows, 4*rows))

        # Flatten the axes array to make it easier to iterate over
        axes = axes.flatten()

        print("Plotting denominators")

        # Loop through each subplot and plot the heatmap
        for i, ax in enumerate(axes):
            if i < rows * rows:
                sns.heatmap(fil[i].detach().to("cpu"), ax = ax, square = True)
            else:
                ax.axis('off')  # Turn off axis for empty subplots

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plot
        plt.savefig(f"plots/denominators/denominator_{btc}.png")


if normed:

    if ckpt:
        covved = voneblockdn.forward(first_sample[0].to(device))
    else:
        covved = outputs_inter / (res + beta)

    for btc in tqdm.tqdm(range(plots_count)):

        rows = int(np.sqrt(sz))
        fil = covved[btc][:sz,::]

        # Create a figure and axis object using Matplotlib
        fig, axes = plt.subplots(rows, rows, figsize=(4*rows, 4*rows))

        # Flatten the axes array to make it easier to iterate over
        axes = axes.flatten()

        print("Plotting norm results")

        # Loop through each subplot and plot the heatmap
        for i, ax in enumerate(axes):
            if i < rows * rows:
                sns.heatmap(fil[i].detach().to("cpu"), ax = ax, square = True)
            else:
                ax.axis('off')  # Turn off axis for empty subplots

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plot
        plt.savefig(f"plots/normed/normed{btc}.png")