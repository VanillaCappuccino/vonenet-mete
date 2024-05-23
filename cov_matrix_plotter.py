import torch
import torchvision
import os,  sys
import os, argparse, time, subprocess, io, shlex, pickle, pprint
import numpy as np
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from vonenet.vonenet import VOneNetDN

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

FLAGS, FIRE_FLAGS = parser.parse_known_args()


output_path = FLAGS.cov_path

cov_matrix = torch.load(output_path+"/cov_matrix.pt")
filters_r = torch.load(output_path+"/real_filters.pt")
filters_c = torch.load(output_path+"/imaginary_filters.pt")

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
                                                shuffle=True,
                                                num_workers=2,
                                                pin_memory=True)

    return data_loader

train_data = data()

for step, dt in enumerate(train_data):

    if step == 0:
        first_sample = dt
        break


# Create a figure and axis object using Matplotlib
cnt = 25
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

vondn = VOneNetDN(simple_channels=simple_channels, gabor_seed=0, complex_channels=complex_channels, model_arch="resnet18", noise_mode = None, k_exc=25, ksize=25, stride = stride, image_size=image_size, visual_degrees=visual_degrees,
                  filters_r = filters_r, filters_c = filters_c, cov_matrix = cov_matrix).to(device)


voneblockdn = vondn[0]

outputs_inter = voneblockdn.gabors_f(first_sample[0].to(device))

sz = outputs_inter.shape[1]
btc = 0 # index of img in batch

for btc in range(25):
    rows = int(np.sqrt(sz))
    fil = outputs_inter[btc][:sz,::]
    print(fil.shape)
    # Create a figure and axis object using Matplotlib
    fig, axes = plt.subplots(rows, rows, figsize=(2*rows, 2*rows))

    # Flatten the axes array to make it easier to iterate over
    axes = axes.flatten()

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

u = np.arange(0,dimm**2).reshape(dimm,dimm)
positions = [u[pos[0]][pos[1]] for pos in inds]



for ind in positions:
    covvies = cov_matrix[ind].reshape(16, image_size//stride, image_size//stride)

    sz = covvies.shape[0]

    rows = int(np.sqrt(sz))
    fil = covvies
    # Create a figure and axis object using Matplotlib
    fig, axes = plt.subplots(rows, rows, figsize=(2*rows, 2*rows))

    # Flatten the axes array to make it easier to iterate over
    axes = axes.flatten()

    # Loop through each subplot and plot the heatmap
    for i, ax in enumerate(axes):
        if i < rows * rows:
            sns.heatmap(fil[i].to("cpu"), ax=ax, cmap="viridis", cbar=False)
        else:
            ax.axis('off')  # Turn off axis for empty subplots

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.savefig(f"cov_matrix_rfs/{ind}.png")


