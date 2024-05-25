import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn
import torchvision.transforms.functional as trn_F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import pickle
# from resnext_50_32x4d import resnext_50_32x4d
# from resnext_101_32x4d import resnext_101_32x4d
# from resnext_101_64x4d import resnext_101_64x4d
from scipy.stats import rankdata
from collections import OrderedDict

import csv
import os, sys
from datetime import datetime
import requests
from tqdm import tqdm
from vonenet import get_model_test, barebones_model, get_dn_model_test

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_built():
    device = "mps"
    mps = True
else:
    device = "cpu"
# import envoy

# dataset = "Tiny-ImageNet-P"

# if dataset == "Tiny-ImageNet-P":
#     url = "https://zenodo.org/records/2536630/files/Tiny-ImageNet-P.tar?download=1"

# dataset = "datasets/"+dataset
# fname = dataset+".tar"

# if not os.path.exists("datasets"):
#     os.mkdir("datasets")


# if not os.path.exists(dataset):

#     if not os.path.exists(dataset+".tar"):
#         # Streaming, so we can iterate over the response.
#         response = requests.get(url, stream=True)

#         # Sizes in bytes.
#         total_size = int(response.headers.get("content-length", 0))
#         block_size = 1024

#         with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
#             with open(fname, "wb") as file:
#                 for data in response.iter_content(block_size):
#                     progress_bar.update(len(data))
#                     file.write(data)

#                 file.close()

#         if total_size != 0 and progress_bar.n != total_size:
#             raise RuntimeError("Could not download file")
        
#     file = open(dataset+".tar", "r")
    
#     if (fname.endswith("tar.gz")):
#         envoy.run("tar xzf %s -C %s" % (file, "datasets"))

#     elif (fname.endswith("tar")):
#         envoy.run("tar xf %s -C %s" % (file, "datasets"))


if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.video_loader import VideoFolder

parser = argparse.ArgumentParser(description='Evaluates robustness of various nets on ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Architecture
parser.add_argument('--model_name', '-m', default='resnet18', type=str,
                    choices=['alexnet', 'squeezenet1.1', 'vgg11', 'vgg19', 'vggbn',
                             'densenet121', 'densenet169', 'densenet201', 'densenet161',
                             'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                             'resnext50', 'resnext101', 'resnext101_64', 'vonenet', 'vonenetdn'])
parser.add_argument('--perturbation', '-p', default='brightness', type=str,
                    choices=['gaussian_noise', 'shot_noise', 'motion_blur', 'zoom_blur',
                             'spatter', 'brightness', 'translate', 'rotate', 'tilt', 'scale',
                             'speckle_noise', 'gaussian_blur', 'snow', 'shear'])
parser.add_argument('--difficulty', '-d', type=int, default=1, choices=[1, 2, 3])
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument("--num_workers", type = int, default = 0, help = "Number of workers.")
parser.add_argument("--rn18_checkpoint", type=str, default="", help = "Location of RN18 checkpoint to load state dict from, if a checkpoint is to be used.")
parser.add_argument("--vonenet_checkpoint", type=str, default="", help = "Location of VOneNet checkpoint to load state dict from, if a checkpoint is to be used.")
parser.add_argument("--vonenetdn_checkpoint", type=str, default="", help = "Location of VOneNetDN checkpoint to load state dict from, if a checkpoint is to be used.")
parser.add_argument("--data_dir", type=str, default="datasets/ImageNet-P", help = "Location of ImageNet-P.")

## Model parameters
parser.add_argument('--torch_seed', default=0, type=int,
                    help='seed for weights initializations and torch RNG')
parser.add_argument('--model_arch', choices=['alexnet', 'resnet18', 'resnet50', 'resnet50_at', 'cornets'], default='resnet18',
                    help='back-end model architecture to load')
parser.add_argument('--normalization', choices=['vonenet', 'imagenet'], default='vonenet',
                    help='image normalization to apply to models')
parser.add_argument('--visual_degrees', default=2, type=float,
                    help='Field-of-View of the model in visual degrees')
parser.add_argument("--model_type", choices = ["barebones", "vonenet", "vonenetdn"], default = "vonenet", help = "Choice of trained model.")

## VOneBlock parameters
# Gabor filter bank
parser.add_argument('--stride', default=2, type=int,
                    help='stride for the first convolution (Gabor Filter Bank)')
parser.add_argument('--ksize', default=25, type=int,
                    help='kernel size for the first convolution (Gabor Filter Bank)')
parser.add_argument('--simple_channels', default=32, type=int,
                    help='number of simple channels in V1 block')
parser.add_argument('--complex_channels', default=32, type=int,
                    help='number of complex channels in V1 block')
parser.add_argument('--gabor_seed', default=0, type=int,
                    help='seed for gabor initialization')
parser.add_argument('--sf_corr', default=0.75, type=float,
                    help='')
parser.add_argument('--sf_max', default=6, type=float,
                    help='')
parser.add_argument('--sf_min', default=0, type=float,
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

parser.add_argument('--cov_path', type = str,
                    help='path to folder that contains cov matrix and filters')

parser.add_argument("--trainable_vonenetdn",  action="store_true", help = "Whether to train divisive norm scalars.")

args = parser.parse_args()
print(args)

visual_degrees = args.visual_degrees
stride = args.stride
ksize = args.ksize
k_exc = args.k_exc
simple_channels = args.simple_channels
complex_channels = args.complex_channels
image_size = 64

# /////////////// Model Setup ///////////////

def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

if args.rn18_checkpoint != "":
    net = barebones_model(model_arch="resnet18", use_TIN = False, imagenet_ckpt=False)
    mdl = torch.load(args.rn18_checkpoint)
    state_dict = mdl["state_dict"]
    print(state_dict)
    net.load_state_dict(state_dict)
    args.test_bs = 5 # value default for rn18.

elif args.vonenet_checkpoint != "":
    mdl = torch.load(args.vonenet_checkpoint)
    print("VOneNet loaded from: ", args.vonenet_checkpoint)

    net = get_model_test()

    ckpts = remove_data_parallel(mdl["state_dict"])

    net.load_state_dict(ckpts)

    args.test_bs = 5 # value default for rn18.

elif args.vonenetdn_checkpoint != "":

    cov_path = args.cov_path

    cov_matrix = torch.load(cov_path+"/cov_matrix.pt").to(device)
    filters_r = torch.load(cov_path+"/real_filters.pt").to(device)
    filters_c = torch.load(cov_path+"/imaginary_filters.pt").to(device)


    mdl = torch.load(args.vonenetdn_checkpoint)
    print("VOneNetDN loaded from: ", args.vonenetdn_checkpoint)

    ckpts = remove_data_parallel(mdl["state_dict"])

    net = get_dn_model_test(map_location=device, pretrained = False,
                                    simple_channels=simple_channels, gabor_seed=0,
                                    complex_channels=complex_channels, model_arch="resnet18", noise_mode = None, k_exc=k_exc, ksize=ksize,
                                    stride = stride, image_size=image_size, visual_degrees=visual_degrees, 
                                    filters_r = filters_r, filters_c = filters_c, cov_matrix = cov_matrix, trainable=args.trainable_vonenetdn)
    args.test_bs = 5 # value default for rn18.

    net.load_state_dict(ckpts)

else:

    if args.model_name == 'alexnet':
        net = models.AlexNet()
        net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
                                            # model_dir='/share/data/lang/users/dan/.torch/models'))
                                            model_dir='/share/data/vision-greg2/pytorch_models/alexnet'))
        args.test_bs = 6

    elif args.model_name == 'resnet18':
        net = models.resnet18()


        net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth',
                                            # model_dir='/share/data/lang/users/dan/.torch/models'))
                                            model_dir='/share/data/vision-greg2/pytorch_models/resnet'))
        args.test_bs = 5

    elif args.model_name == 'resnet34':
        net = models.resnet34()
        net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth',
                                            # model_dir='/share/data/lang/users/dan/.torch/models'))
                                            model_dir='/share/data/vision-greg2/pytorch_models/resnet'))
        args.test_bs = 4

    elif args.model_name == 'resnet50':
        net = models.resnet50()
        net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth',
                                            # model_dir='/share/data/lang/users/dan/.torch/models'))
                                            model_dir='/share/data/vision-greg2/pytorch_models/resnet'))
        args.test_bs = 4

    elif args.model_name == 'resnet101':
        net = models.resnet101()
        net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
                                            # model_dir='/share/data/lang/users/dan/.torch/models'))
                                            model_dir='/share/data/vision-greg2/pytorch_models/resnet'))
        args.test_bs = 3

args.prefetch = 4

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

print('Model Loaded\n')

# /////////////// Data Loader ///////////////
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

num_workers = args.num_workers

root_dir = args.data_dir + "/"

if args.difficulty > 1 and 'noise' in args.perturbation:
    loader = torch.utils.data.DataLoader(
        VideoFolder(root=root_dir +
                         args.perturbation + '_' + str(args.difficulty),
                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])),
        batch_size=args.test_bs, shuffle=False, num_workers=num_workers, pin_memory=True)
else:
    loader = torch.utils.data.DataLoader(
        VideoFolder(root=root_dir + args.perturbation,
                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])),
        batch_size=args.test_bs, shuffle=False, num_workers=num_workers, pin_memory=True)

print('Data Loaded\n')


# /////////////// Stability Measurements ///////////////

identity = np.asarray(range(1, 201))
cum_sum_top5 = np.cumsum(np.asarray([0] + [1] * 5 + [0] * (199 - 5)))
recip = 1./identity

# def top5_dist(sigma):
#     result = 0
#     for i in range(1,6):
#         for j in range(min(sigma[i-1], i) + 1, max(sigma[i-1], i) + 1):
#             if 1 <= j - 1 <= 5:
#                 result += 1
#     return result

def dist(sigma, mode='top5'):
    if mode == 'top5':
        return np.sum(np.abs(cum_sum_top5[:5] - cum_sum_top5[sigma-1][:5]))
    elif mode == 'zipf':
        return np.sum(np.abs(recip - recip[sigma-1])*recip)


def ranking_dist(ranks, noise_perturbation=True if 'noise' in args.perturbation else False, mode='top5'):
    result = 0
    step_size = 1 if noise_perturbation else args.difficulty

    for vid_ranks in ranks:
        result_for_vid = []

        for i in range(step_size):
            perm1 = vid_ranks[i]
            perm1_inv = np.argsort(perm1)

            for rank in vid_ranks[i::step_size][1:]:
                perm2 = rank
                result_for_vid.append(dist(perm2[perm1_inv], mode))
                if not noise_perturbation:
                    perm1 = perm2
                    perm1_inv = np.argsort(perm1)

        result += np.mean(result_for_vid) / len(ranks)

    return result


def flip_prob(predictions, noise_perturbation=True if 'noise' in args.perturbation else False):
    result = 0
    step_size = 1 if noise_perturbation else args.difficulty

    for vid_preds in predictions:
        result_for_vid = []

        for i in range(step_size):
            prev_pred = vid_preds[i]

            for pred in vid_preds[i::step_size][1:]:
                result_for_vid.append(int(prev_pred != pred))
                if not noise_perturbation: prev_pred = pred

        result += np.mean(result_for_vid) / len(predictions)

    return result


# /////////////// Get Results ///////////////

from tqdm import tqdm

if "Tiny" in root_dir:
    dim = 64
else:
    dim = 224


predictions, ranks = [], []
with torch.no_grad():

    for data, target in tqdm(loader):
        num_vids = data.size(0)
        data = data.view(-1,3,dim,dim).cuda()

        # print(data.shape)

        output = net(data)

        # print(output.shape)

        for vid in output.view(num_vids, -1, 200):
            predictions.append(vid.argmax(1).to('cpu').numpy())
            ranks.append([np.uint16(rankdata(-frame, method='ordinal')) for frame in vid.to('cpu').numpy()])


ranks = np.asarray(ranks)

filename = "results.csv"
model_name = args.model_name

date_time = datetime.now().strftime("%d/%m/%Y|%H:%M:%S")

contents = {"Model Name": [], "RunID": [], "Perturbation": [], "FR": [], "T5D": [], "Zipf": []}
field_names = list(contents.keys())

if not os.path.exists(filename):

    with open(filename, "a+") as f:
        w = csv.DictWriter(f, field_names)
        w.writeheader()
        # w.writerows(contents)

    f.close()

print('Computing Metrics\n')

mfr = flip_prob(predictions)
mt5 = ranking_dist(ranks, mode='top5')
zipf = ranking_dist(ranks, mode='zipf')

print('Flipping Prob\t{:.5f}'.format(mfr))
print('Top5 Distance\t{:.5f}'.format(mt5))
print('Zipf Distance\t{:.5f}'.format(zipf))

if args.difficulty > 1 and "noise" in args.perturbation:
    prt = args.perturbation + args.difficulty
else:
    prt = args.perturbation

records = []
contents = [{"Model Name": model_name, "RunID": date_time, "Perturbation": prt, "mFR": "{:.5f}".format(mfr), "mT5D": "{:.5f}".format(mt5), "Zipf": "{:.5f}".format(zipf)}]

try:
    results_old = pickle.load(open(os.path.join("", 'results.pkl'), 'rb'))
except:
    results_old = []
    pass

for entry in results_old:
    records.append(entry)

records.append(contents)
pickle.dump(records, open(os.path.join("", 'results.pkl'), 'wb'))
