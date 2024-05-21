# -*- coding: utf-8 -*-

import argparse
import os
import time
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as trn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import numpy as np
from vonenet import get_model_test, barebones_model, get_dn_model_test
import pickle
from datetime import datetime
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Evaluates robustness of various nets on ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Architecture
parser.add_argument('--model_arch', '-m', type=str,
                    choices=['alexnet',
                             'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                             'resnext50', 'resnext101', 'resnext101_64'])
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--data_dir', type=str, default="datasets/ImageNet-C", help="Dataset directory.")

parser.add_argument("--rn18_checkpoint", type=str, default="", help = "Location of RN18 checkpoint to load state dict from, if a checkpoint is to be used.")
parser.add_argument("--vonenet_checkpoint", type=str, default="", help = "Location of VOneNet checkpoint to load state dict from, if a checkpoint is to be used.")
parser.add_argument("--vonenetdn_checkpoint", type=str, default="", help = "Location of VOneNetDN checkpoint to load state dict from, if a checkpoint is to be used.")

parser.add_argument("--identifier", type=str, default="")

args = parser.parse_args()
print(args)

# /////////////// Model Setup ///////////////

if args.rn18_checkpoint != "":
    model_type = "resnet18"
    net = barebones_model(model_arch="resnet18", use_TIN = False, imagenet_ckpt=False)
    mdl = torch.load(args.rn18_checkpoint)
    state_dict = mdl["state_dict"]

    net.load_state_dict(state_dict)
    args.test_bs = 5 # value default for rn18.

elif args.vonenet_checkpoint != "":
    model_type = "vonenet"
    mdl = torch.load(args.vonenet_checkpoint)
    
    net = get_model_test(mdl, "resnet18", "cpu")

    args.test_bs = 5 # value default for rn18.

elif args.vonenetdn_checkpoint != "":
    model_type = "vonenetdn"

    mdl = torch.load(args.vonenetdn_checkpoint)

    net = get_dn_model_test(mdl, "resnet18", "cpu")

    args.test_bs = 5 # value default for rn18.

elif args.model_name == 'alexnet':
    net = models.AlexNet()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
                                           model_dir='/share/data/lang/users/dan/.torch/models'))
    args.test_bs = 256


elif args.model_name == 'resnet18':
    net = models.resnet18()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth',
                                           model_dir='/share/data/lang/users/dan/.torch/models'))
    args.test_bs = 256

elif args.model_name == 'resnet34':
    net = models.resnet34()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth',
                                           model_dir='/share/data/lang/users/dan/.torch/models'))
    args.test_bs = 128

elif args.model_name == 'resnet50':
    net = models.resnet50()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth',
                                           model_dir='/share/data/lang/users/dan/.torch/models'))
    args.test_bs = 128

elif args.model_name == 'resnet101':
    net = models.resnet101()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
                                           model_dir='/share/data/lang/users/dan/.torch/models'))
    args.test_bs = 128

elif args.model_name == 'resnet152':
    net = models.resnet152()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet152-b121ed2d.pth',
                                           model_dir='/share/data/lang/users/dan/.torch/models'))
    args.test_bs = 64

args.prefetch = 4

for p in net.parameters():
    p.volatile = True

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()

torch.manual_seed(1)
np.random.seed(1)
if args.ngpu > 0:
    torch.cuda.manual_seed(1)

net.eval()
cudnn.benchmark = True  # fire on all cylinders

print('Model Loaded')

# /////////////// Data Loader ///////////////

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
# clean_loader = torch.utils.data.DataLoader(dset.ImageFolder(
#     root="/share/data/vision-greg/ImageNet/clsloc/images/val",
#     transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])),
#     batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)


# /////////////// Further Setup ///////////////

def auc(errs):  # area under the distortion-error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area


# correct = 0
# for batch_idx, (data, target) in enumerate(clean_loader):
#     data = V(data.cuda(), volatile=True)
#
#     output = net(data)
#
#     pred = output.data.max(1)[1]
#     correct += pred.eq(target.cuda()).sum()
#
# clean_error = 1 - correct / len(clean_loader.dataset)
# print('Clean dataset error (%): {:.2f}'.format(100 * clean_error))


def show_performance(distortion_name):
    errs = []

    for severity in tqdm(range(1, 6)):
        distorted_dataset = dset.ImageFolder(
            root=args.data_dir + '/' + distortion_name + '/' + str(severity),
            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))

        distorted_dataset_loader = torch.utils.data.DataLoader(
            distorted_dataset, batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)

        correct = 0
        for batch_idx, (data, target) in tqdm(enumerate(distorted_dataset_loader)):
            data = V(data.cuda(), volatile=True)

            output = net(data)

            pred = output.data.max(1)[1]
            correct += pred.eq(target.cuda()).sum()

        errs.append(np.float64(1 - 1.*correct / len(distorted_dataset)))

    print('\n=Average', tuple(errs))
    return np.mean(errs), errs


# /////////////// End Further Setup ///////////////


# /////////////// Display Results ///////////////
import collections

print('\nUsing ImageNet data')

distortions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

# 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'

error_rates = []
contents = []

for distortion_name in distortions:
    rate, errs = show_performance(distortion_name)
    error_rates.append(rate)
    contents.append({"distortion": distortion_name, "error_rate": rate, "errs": errs})
    print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion_name, 100 * rate))


print('mCE (unnormalized by AlexNet errors) (%): {:.2f}'.format(100 * np.mean(error_rates)))

records = []

try:
    results_old = pickle.load(open(os.path.join("", 'imagenet_c_results.pkl'), 'rb'))
except:
    results_old = []
    pass

for entry in results_old:
    records.append(entry)

cnts = dict()
cnts["meta"] = {"test": model_type + "_" + args.identifier, "time": datetime.now().strftime("%d/%m/%Y|%H:%M:%S")}
cnts["results"] = contents

records.append(cnts)

pickle.dump(records, open(os.path.join("", 'imagenet_c_results.pkl'), 'wb'))