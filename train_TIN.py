import os, argparse, time, subprocess, io, shlex, pickle, pprint
import pandas as pd
import numpy as np
import tqdm
import fire

import torch
import torch.nn as nn
import torch.utils.model_zoo
import torchvision
from vonenet import get_model, barebones_model, get_dn_model


torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='Tiny ImageNet Training')
## General parameters
parser.add_argument('--in_path', required=True,
                    help='path to ImageNet folder that contains train and val folders')
parser.add_argument('--cov_path', type = str,
                    help='path to folder that contains cov matrix and filters')

parser.add_argument('-o', '--output_path', default=None,
                    help='path for storing ')
parser.add_argument('-restore_epoch', '--restore_epoch', default=0, type=int,
                    help='epoch number for restoring model training ')
parser.add_argument('-restore_path', '--restore_path', default=None, type=str,
                    help='path of folder containing specific epoch file for restoring model training')

## Training parameters
parser.add_argument('--ngpus', default=0, type=int,
                    help='number of GPUs to use; 0 if you want to run on CPU')
parser.add_argument('-j', '--workers', default=20, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=60, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int,
                    help='mini-batch size')
parser.add_argument('--optimizer', choices=['stepLR', 'plateauLR', 'adam'], default='plateauLR',
                    help='Optimizer')
parser.add_argument('--lr', '--learning_rate', default=.1, type=float,
                    help='initial learning rate')
parser.add_argument('--step_size', default=5, type=int,
                    help='after how many epochs learning rate should be decreased by step_factor')
parser.add_argument('--step_factor', default=0.1, type=float,
                    help='factor by which to decrease the learning rate')
parser.add_argument('--momentum', default=.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='weight decay ')
parser.add_argument("--save_model_epochs", default=1, type=int, help="Overwrite or record epoch weights separately")
parser.add_argument("--latest_only", action="store_true", help="Save each stepped epoch or only the latest corresponding to frequency")
parser.add_argument("--imagenet1k_ckpt", action="store_true", help="Use ImageNet1k checkpoint from Torchvision.")

## Model parameters
parser.add_argument("--image_size", default = 64, type = int, help = "Input image size.")
parser.add_argument('--torch_seed', default=0, type=int,
                    help='seed for weights initializations and torch RNG')
parser.add_argument('--model_arch', choices=['alexnet', 'resnet18', 'resnet50', 'resnet50_at', 'cornets'], default='resnet18',
                    help='back-end model architecture to load')
parser.add_argument('--normalization', choices=['vonenet', 'imagenet'], default='vonenet',
                    help='image normalization to apply to models')
parser.add_argument('--visual_degrees', default=2, type=float,
                    help='Field-of-View of the model in visual degrees')
parser.add_argument("--model_type", choices = ["barebones", "vonenet", "vonenetdn"], default = "vonenet", help = "Choice of trained model.")
parser.add_argument("--trainable_vonenetdn",  action="store_true", help = "Whether to train divisive norm scalars.")
parser.add_argument("--paper_implementation", action="store_true", help = "Use the implementation of DN in the original paper.")



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


FLAGS, FIRE_FLAGS = parser.parse_known_args()

use_TIN = True

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

# if FLAGS.overwrite:
#     overwrite=None
# else:
#     overwrite=1

torch.manual_seed(FLAGS.torch_seed)

torch.backends.cudnn.benchmark = True

mps = False

# if FLAGS.ngpus > 0:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# else:
#     device = 'cpu'

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_built():
    device = "mps"
    mps = True
else:
    device = "cpu"

print("Device: ", device)

simchan = FLAGS.simple_channels
comchan = FLAGS.complex_channels
batch_size = FLAGS.batch_size

cov_dir = f"{simchan}x{comchan}"


file_dir = cov_dir + "x" + str(batch_size)

print("Cov matrix directory: ", file_dir)

cov_matrix = None
filters_r = None
filters_c = None


if FLAGS.model_type == "vonenetdn" and not FLAGS.paper_implementation:

    print("Loading covariance structures.")

    if os.path.exists(FLAGS.cov_path):
        print("Cov matrix directory: ", FLAGS.cov_path)
        cov_matrix = torch.load(f"{FLAGS.cov_path}/cov_matrix.pt", map_location = device)
        filters_r = torch.load(f"{FLAGS.cov_path}/real_filters.pt", map_location = device)
        filters_c = torch.load(f"{FLAGS.cov_path}/imaginary_filters.pt", map_location = device)

    elif os.path.exists(file_dir):
        print("Cov matrix directory: ", file_dir)
        cov_matrix = torch.load(f"{file_dir}/cov_matrix.pt", map_location = device)
        filters_r = torch.load(f"{file_dir}/real_filters.pt", map_location = device)
        filters_c = torch.load(f"{file_dir}/imaginary_filters.pt", map_location = device)

    elif os.path.exists(cov_dir):
        print("Cov matrix directory: ", cov_dir)
        cov_matrix = torch.load(f"{cov_dir}/cov_matrix.pt", map_location = device)
        filters_r = torch.load(f"{cov_dir}/real_filters.pt", map_location = device)
        filters_c = torch.load(f"{cov_dir}/imaginary_filters.pt", map_location = device)

    else:
        raise ValueError("There exist no pre-trained covariance values for this divisive normalisation configuration.")


if FLAGS.normalization == 'vonenet':
    print('VOneNet normalization')
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
elif FLAGS.normalization == 'imagenet':
    print('Imagenet standard normalization')
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    
map_location = None if FLAGS.ngpus > 0 else (device if mps else 'cpu')

def load_model():

    print('Getting VOneNet')

    if FLAGS.model_type == "barebones":
        model = barebones_model(model_arch=FLAGS.model_arch, use_TIN = use_TIN, imagenet_ckpt = FLAGS.imagenet1k_ckpt)
    elif FLAGS.model_type == "vonenetdn":
        model = get_dn_model(map_location=map_location, model_arch=FLAGS.model_arch, pretrained=False,
                visual_degrees=FLAGS.visual_degrees, stride=FLAGS.stride, ksize=FLAGS.ksize,
                sf_corr=FLAGS.sf_corr, sf_max=FLAGS.sf_max, sf_min=FLAGS.sf_min, rand_param=FLAGS.rand_param,
                gabor_seed=FLAGS.gabor_seed, simple_channels=FLAGS.simple_channels,
                complex_channels=FLAGS.simple_channels, noise_mode=FLAGS.noise_mode,
                noise_scale=FLAGS.noise_scale, noise_level=FLAGS.noise_level, k_exc=FLAGS.k_exc,
                cov_matrix = cov_matrix, filters_r = filters_r, filters_c = filters_c, trainable = FLAGS.trainable_vonenetdn,
                paper_implementation = FLAGS.paper_implementation, image_size = FLAGS.image_size)
    else:
        model = get_model(map_location=map_location, model_arch=FLAGS.model_arch, pretrained=False,
                      visual_degrees=FLAGS.visual_degrees, stride=FLAGS.stride, ksize=FLAGS.ksize,
                      sf_corr=FLAGS.sf_corr, sf_max=FLAGS.sf_max, sf_min=FLAGS.sf_min, rand_param=FLAGS.rand_param,
                      gabor_seed=FLAGS.gabor_seed, simple_channels=FLAGS.simple_channels,
                      complex_channels=FLAGS.simple_channels, noise_mode=FLAGS.noise_mode,
                      noise_scale=FLAGS.noise_scale, noise_level=FLAGS.noise_level, k_exc=FLAGS.k_exc, use_TIN = use_TIN,
                      image_size = FLAGS.image_size)

    if FLAGS.ngpus > 0 and torch.cuda.device_count() > 1:
        print('We have multiple GPUs detected')
        model = model.to(device)
    elif FLAGS.ngpus > 0 and torch.cuda.device_count() == 1:
        print('We run on GPU')
        model = model.to(device)
    elif mps:
        print("Metal hardware acceleration")
        model = model.to(device)
    else:
        print('No GPU detected!')
        model = model.module

    return model


def train(save_train_epochs=.2,  # how often save output during training
          save_val_epochs=.5,  # how often save output during validation
          save_model_epochs=FLAGS.save_model_epochs,  # how often save model weights
          save_model_secs=720 * 10  # how often save model (in sec)
          ):
    
    model = load_model()

    trainer = ImageNetTrain(model)
    validator = ImageNetVal(model)

    start_epoch = 0
    records = []

    if FLAGS.restore_epoch > 0:
        print('Restoring from previous...')
        ckpt_data = torch.load(os.path.join(FLAGS.restore_path, f'epoch_{FLAGS.restore_epoch:02d}.pth.tar'))
        start_epoch = ckpt_data['epoch']
        print('Loaded epoch: '+str(start_epoch))
        model.load_state_dict(ckpt_data['state_dict'])
        trainer.optimizer.load_state_dict(ckpt_data['optimizer'])
        results_old = pickle.load(open(os.path.join(FLAGS.restore_path, 'results.pkl'), 'rb'))
        for result in results_old:
            records.append(result)

    results = {'meta': {'step_in_epoch': 0,
                        'epoch': start_epoch,
                        'wall_time': time.time()}
               }

    # records = []
    recent_time = time.time()

    nsteps = len(trainer.data_loader)

    if save_train_epochs is not None:
        save_train_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_train_epochs) * nsteps).astype(int)
    if save_val_epochs is not None:
        save_val_steps = (np.arange(0, FLAGS.epochs + 1,
                                    save_val_epochs) * nsteps).astype(int)
    if save_model_epochs is not None:
        save_model_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_model_epochs) * nsteps).astype(int)

    for epoch in tqdm.trange(start_epoch, FLAGS.epochs + 1, initial=0, desc='epoch'):
        print(epoch)
        data_load_start = np.nan

        data_loader_iter = trainer.data_loader

        for step, data in enumerate(tqdm.tqdm(data_loader_iter, desc=trainer.name)):
            data_load_time = time.time() - data_load_start
            global_step = epoch * nsteps + step

            if save_val_steps is not None:
                if global_step in save_val_steps:
                    results[validator.name] = validator()
                    if FLAGS.optimizer == 'plateauLR' and step == 0:
                        trainer.lr.step(results[validator.name]['loss'])
                    trainer.model.train()

                    # if FLAGS.paper_implementation:
                    #     trainer.model[0].hidden_beta.detach()
                    #     trainer.model[0].hidden_params.detach()

                    print('LR: ', trainer.optimizer.param_groups[0]["lr"])

            if FLAGS.output_path is not None:
                if not (os.path.isdir(FLAGS.output_path)):
                    os.mkdir(FLAGS.output_path)

                records.append(results)
                if len(results) > 1:
                    pickle.dump(records, open(os.path.join(FLAGS.output_path, 'results.pkl'), 'wb'))

                ckpt_data = {}
                ckpt_data['flags'] = FLAGS.__dict__.copy()
                ckpt_data['epoch'] = epoch
                ckpt_data['state_dict'] = model.state_dict()
                ckpt_data['optimizer'] = trainer.optimizer.state_dict()

                if save_model_secs is not None:
                    if time.time() - recent_time > save_model_secs:
                        torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                           'latest_checkpoint.pth.tar'))
                        recent_time = time.time()

                if save_model_steps is not None:
                    if global_step in save_model_steps:
                        if FLAGS.latest_only:
                            torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                        f'latest_epoch.pth.tar'))
                        else:
                            torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                            f'epoch_{epoch:02d}.pth.tar'))

            else:
                if len(results) > 1:
                    pprint.pprint(results)

            if epoch < FLAGS.epochs:
                frac_epoch = (global_step + 1) / nsteps
                record = trainer(frac_epoch, *data)
                record['data_load_dur'] = data_load_time
                results = {'meta': {'step_in_epoch': step + 1,
                                    'epoch': frac_epoch,
                                    'wall_time': time.time()}
                           }
                if save_train_steps is not None:
                    if step in save_train_steps:
                        results[trainer.name] = record

            data_load_start = time.time()


class ImageNetTrain(object):

    def __init__(self, model):
        self.name = 'train'
        self.model = model
        self.data_loader = self.data()

        if FLAGS.optimizer != "adam":
            self.optimizer = torch.optim.SGD(self.model.parameters(), FLAGS.lr, momentum=FLAGS.momentum,
                                            weight_decay=FLAGS.weight_decay)
            if FLAGS.optimizer == 'stepLR':
                self.lr = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=FLAGS.step_factor,
                                                        step_size=FLAGS.step_size)
            elif FLAGS.optimizer == 'plateauLR':
                self.lr = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=FLAGS.step_factor,
                                                                    patience=FLAGS.step_size-1, threshold=0.01)

            
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=FLAGS.lr) 



        self.loss = nn.CrossEntropyLoss()
        if FLAGS.ngpus > 0:
            self.loss = self.loss.cuda()

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.in_path, 'train'),
            torchvision.transforms.Compose([
                torchvision.transforms.RandomAffine(degrees=30, translate=(0.05, 0.05), scale=(1, 1.2)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=norm_mean, std=norm_std)
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=True,
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True)

        return data_loader

    def __call__(self, frac_epoch, inp, target):
        start = time.time()
        if FLAGS.optimizer == 'stepLR':
            self.lr.step(epoch=frac_epoch)
        target = target.to(device)

        # inp = inp.to(device)

        output = self.model(inp)

        record = {}
        loss = self.loss(output, target)
        record['loss'] = loss.item()
        record['top1'], record['top5'] = accuracy(output, target, topk=(1, 5))
        record['top1'] /= len(output)
        record['top5'] /= len(output)
        # record['learning_rate'] = self.lr.get_lr()[0]
        record['learning_rate'] = self.optimizer.param_groups[0]["lr"]
        self.optimizer.zero_grad()
        loss.backward(retain_graph = True)
        self.optimizer.step()

        record['dur'] = time.time() - start
        return record


# evaluation class

class ImageNetVal(object):

    def __init__(self, model):
        self.name = 'val'
        self.model = model
        self.data_loader = self.data()
        self.loss = nn.CrossEntropyLoss(size_average=False)
        self.loss = self.loss.to(device)

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.in_path, 'val'),
            torchvision.transforms.Compose([
                # torchvision.transforms.Resize(256),
                # torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=norm_mean, std=norm_std),
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=False,
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True)

        return data_loader

    def __call__(self):
        self.model.eval()
        start = time.time()
        record = {'loss': 0, 'top1': 0, 'top5': 0}
        with torch.no_grad():
            for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):
                
                # inp = inp.to(device)
                target = target.to(device)
                output = self.model(inp)

                record['loss'] += self.loss(output, target).item()
                p1, p5 = accuracy(output, target, topk=(1, 5))
                record['top1'] += p1
                record['top5'] += p5

        for key in record:
            record[key] /= len(self.data_loader.dataset.samples)
        record['dur'] = (time.time() - start) / len(self.data_loader)

        return record


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res


if __name__ == '__main__':
    fire.Fire(command=FIRE_FLAGS)
