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


vondn = VOneNetDN(simple_channels=simple_channels, gabor_seed=0, complex_channels=complex_channels, model_arch="resnet18", noise_mode = None, k_exc=25, ksize=25, stride = stride, image_size=image_size, visual_degrees=visual_degrees,
                  filters_r = filters_r, filters_c = filters_c, cov_matrix = cov_matrix).to(device)


