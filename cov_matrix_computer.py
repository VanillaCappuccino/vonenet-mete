import torch
import torchvision
import numpy as np
import tqdm
import seaborn as sns
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

von = VOneNet(simple_channels=simple_channels, complex_channels=complex_channels, model_arch="resnet18", k_exc=k_exc, ksize=ksize, stride = stride, image_size=image_size, visual_degrees=visual_degrees)

voneblock = von[0]

filters_r = voneblock.simple_conv_q0
filters_c = voneblock.simple_conv_q1

output_path = FLAGS.output_path
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

cov_dim = batch_size*image_size**2
cov_matrix = torch.zeros(cov_dim, cov_dim)

for step, data in enumerate(tqdm.tqdm(train_data)):
    
    outputs = voneblock.forward(data[0])

    p1 = outputs.reshape(batch_size,-1)
    term1 = p1.T @ p1 / data[1].shape[0]

    m1 = torch.mean(p1, dim=0)
    m1.shape
    mn = torch.outer(m1, m1)

    cov_matrix += term1 - mn

cov_matrix /= count

torch.save(cov_matrix, output_path+"/cov_matrix.pt")