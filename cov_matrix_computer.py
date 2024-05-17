import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import numpy as np
import tqdm
# import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os, sys, argparse, time, subprocess, io, shlex, pickle, pprint
import pandas as pd
import fire
import cv2

from vonenet.vonenet import VOneNet

parser = argparse.ArgumentParser(description='ImageNet Training')
## General parameters
parser.add_argument('--in_path', required=True,
                    help='path to ImageNet folder that contains train and val folders')
parser.add_argument('--out_path', required=True,
                    help='path to ImageNet folder that contains train and val folders')
parser.add_argument('--model_arch', choices=['alexnet', 'resnet18', 'resnet50', 'resnet50_at', 'cornets'], default='resnet18',
                    help='back-end model architecture to load')
parser.add_argument('--normalization', choices=['vonenet', 'imagenet'], default='vonenet',
                    help='image normalization to apply to models')
parser.add_argument('--batch_size', default=128, type=int,
                    help='mini-batch size')
parser.add_argument('--stride', default=2, type=int,
                    help='stride for the first convolution (Gabor Filter Bank)')
parser.add_argument('--visual_degrees', default=2, type=float,
                    help='Field-of-View of the model in visual degrees')
parser.add_argument('--ksize', default=7, type=int,
                    help='kernel size for the first convolution (Gabor Filter Bank)')
parser.add_argument('--simple_channels', default=16, type=int,
                    help='number of simple channels in V1 block')
parser.add_argument('--complex_channels', default=16, type=int,
                    help='number of complex channels in V1 block')
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                    help='Device to use for computation')

FLAGS, FIRE_FLAGS = parser.parse_known_args()

in_path = FLAGS.in_path
model_arch = FLAGS.model_arch
normalization = FLAGS.normalization
batch_size = FLAGS.batch_size
stride = FLAGS.stride
colour = 3
visual_degrees = FLAGS.visual_degrees
ksize = FLAGS.ksize
k_exc = ksize
simple_channels = FLAGS.simple_channels
complex_channels = FLAGS.complex_channels
image_size = 64

if normalization == 'vonenet':
    print('VOneNet normalization')
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
elif normalization == 'imagenet':
    print('Imagenet standard normalization')
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

if FLAGS.device == "cuda" and torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_built():
    device = "mps"
    mps = True
else:
    device = "cpu"

print("Device: ", device)

von = VOneNet(simple_channels=simple_channels, complex_channels=complex_channels, model_arch="resnet18", noise_mode = None, k_exc=25, ksize=25, stride = stride, image_size=image_size, visual_degrees=visual_degrees).to(device)

voneblock = von[0]

ksz = (ksize,ksize)

voneblock.simple_conv_q0.weight = nn.Parameter(F.interpolate(voneblock.simple_conv_q0.weight, size = ksz), requires_grad = False)
voneblock.simple_conv_q1.weight = nn.Parameter(F.interpolate(voneblock.simple_conv_q1.weight, size = ksz), requires_grad = False)

voneblock.ksize = ksize

voneblock.simple_conv_q0.padding = (voneblock.ksize//2, voneblock.ksize//2)
voneblock.simple_conv_q1.padding = (voneblock.ksize//2, voneblock.ksize//2)

filters_r = voneblock.simple_conv_q0
filters_c = voneblock.simple_conv_q1

output_path = FLAGS.out_path
try:
    os.mkdir(output_path)
except:
    print("Directory already exists.")
    pass

torch.save(filters_r, output_path+"/real_filters.pt")
torch.save(filters_c, output_path+"/imaginary_filters.pt")


def data():
    dataset = torchvision.datasets.ImageFolder(
        os.path.join("tiny-imagenet-200", 'train'),
        torchvision.transforms.Compose([
            # torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomAffine(degrees=30, translate=(0.05, 0.05), scale=(1, 1.2)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=norm_mean, std=norm_std)
        ]))
    data_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=2,
                                                pin_memory=True)

    return data_loader

train_data = data()
count = len(train_data)

cov_dim = (simple_channels+complex_channels)*32**2
print(cov_dim)
cov_matrix = torch.zeros(cov_dim, cov_dim).to(device)

for step, data in enumerate(tqdm.tqdm(train_data)):
    
    with torch.no_grad():
        
        outputs = voneblock.forward(data[0].to(device))

        if device == "cuda":

            print(torch.cuda.mem_get_info())

        p1 = outputs.reshape(-1, cov_dim)

        if device == "cuda":

            print(torch.cuda.mem_get_info())

        term1 = p1.T @ p1 / data[1].shape[0]

        if device == "cuda":

            print(torch.cuda.mem_get_info())

        m1 = torch.mean(p1, dim=0)
        m1.shape
        mn = torch.outer(m1, m1)

        cov_matrix += term1 - mn

        torch.cuda.empty_cache()

    print(outputs.device, p1.device, m1.device, mn.device, cov_matrix.device)


cov_matrix /= count

torch.save(cov_matrix, output_path+"/cov_matrix.pt")