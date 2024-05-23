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

from vonenet import get_model
from vonenet.vonenet import VOneNet

parser = argparse.ArgumentParser(description='ImageNet Training')
## General parameters
parser.add_argument('--in_path', required=True,
                    help='path to ImageNet folder that contains train and val folders')
parser.add_argument('--out_path', required=True,
                    help='path to ImageNet folder that contains train and val folders')
parser.add_argument('--model_arch', choices=['alexnet', 'resnet18', 'resnet50', 'resnet50_at', 'cornets'], default='resnet18',
                    help='back-end model architecture to load')
parser.add_argument('--ngpus', default=0, type=int,
                    help='number of GPUs to use; 0 if you want to run on CPU')

parser.add_argument('--normalization', choices=['vonenet', 'imagenet'], default='vonenet',
                    help='image normalization to apply to models')
parser.add_argument('--batch_size', default=128, type=int,
                    help='mini-batch size')
parser.add_argument('--stride', default=2, type=int,
                    help='stride for the first convolution (Gabor Filter Bank)')
parser.add_argument('--visual_degrees', default=2, type=float,
                    help='Field-of-View of the model in visual degrees')
parser.add_argument('--ksize', default=25, type=int,
                    help='kernel size for the first convolution (Gabor Filter Bank)')
parser.add_argument('--simple_channels', default=32, type=int,
                    help='number of simple channels in V1 block')
parser.add_argument('--complex_channels', default=32, type=int,
                    help='number of complex channels in V1 block')
parser.add_argument('--gabor_seed', default=0, type=int,
                    help='seed for gabor initialization')

parser.add_argument('--rgb_seed', default=0, type=int,
                    help='seed for gabor initialization')

parser.add_argument('--image_size', default=64, type=int,
                    help='seed for gabor initialization')
parser.add_argument('--sf_corr', default=0.75, type=float,
                    help='')
parser.add_argument('--sf_max', default=11.2, type=float,
                    help='')
parser.add_argument('--sf_min', default=0.5, type=float,
                    help='')
parser.add_argument('--rand_param', choices=[True, False], default=False, type=bool,
                    help='random gabor params')
parser.add_argument('--k_exc', default=25, type=float,
                    help='')

# Noise layer
parser.add_argument('--noise_mode', choices=['gaussian', 'neuronal', None],
                    default=None,
                    help='noise distribution')
parser.add_argument('--noise_scale', default=1, type=float,
                    help='noise scale factor')
parser.add_argument('--noise_level', default=1, type=float,
                    help='noise level')



parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu',
                    help='Device to use for computation')
parser.add_argument('--torch_seed', default=0, type=int,
                    help='seed for weights initializations and torch RNG')

FLAGS, FIRE_FLAGS = parser.parse_known_args()

def set_gpus(n=2):
    """
    Finds all GPUs on the system and restricts to n of them that have the most
    free memory.
    """
    if n > 0:
        gpus = subprocess.run(shlex.split(
            'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True,
            stdout=subprocess.PIPE).stdout
        gpus = pd.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
        gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
        if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
            visible = [int(i)
                       for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
            gpus = gpus[gpus['index'].isin(visible)]
        gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # making sure GPUs are numbered the same way as in nvidia_smi
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            [str(i) for i in gpus['index'].iloc[:n]])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if FLAGS.ngpus > 0:
    print("Setting GPUs.")
    set_gpus(FLAGS.ngpus)

torch.manual_seed(FLAGS.torch_seed)

torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_built():
    device = "mps"
    mps = True
else:
    device = "cpu"

map_location = None if FLAGS.ngpus > 0 else (device if mps else 'cpu')

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
image_size = FLAGS.image_size
rgb_seed = FLAGS.rgb_seed

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
elif FLAGS.device == "mps" and torch.backends.mps.is_built():
    device = "mps"
    mps = True
else:
    device = "cpu"

print("Device: ", device)

von = VOneNet(model_arch=FLAGS.model_arch, pretrained=False,
                visual_degrees=FLAGS.visual_degrees, stride=FLAGS.stride, ksize=FLAGS.ksize,
                sf_corr=FLAGS.sf_corr, sf_max=FLAGS.sf_max, sf_min=FLAGS.sf_min, rand_param=FLAGS.rand_param,
                gabor_seed=FLAGS.gabor_seed, simple_channels=FLAGS.simple_channels,
                complex_channels=FLAGS.simple_channels, noise_mode=FLAGS.noise_mode,
                noise_scale=FLAGS.noise_scale, noise_level=FLAGS.noise_level, k_exc=FLAGS.k_exc, use_TIN = True,
                image_size = image_size, rgb_seed = rgb_seed)

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
                                                num_workers=16,
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

        p1 = outputs.reshape(-1, cov_dim)

        term1 = p1.T @ p1 / data[1].shape[0]

        m1 = torch.mean(p1, dim=0)
        m1.shape
        mn = torch.outer(m1, m1)

        cov_matrix += term1 - mn

        if device == "cuda":
            torch.cuda.empty_cache()


cov_matrix /= count

torch.save(cov_matrix, output_path+"/cov_matrix.pt")