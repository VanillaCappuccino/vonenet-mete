
from collections import OrderedDict
from torch import nn
from .modules import VOneBlock, VOneBlockDN
from .back_ends import ResNetBackEnd, BasicBlock, Bottleneck, AlexNetBackEnd, CORnetSBackEnd
from .params import generate_gabor_param
import numpy as np
from torch.nn import functional as F


def VOneNetDN(sf_corr=0.75, sf_max=9, sf_min=0, rand_param=False, gabor_seed=0,
            simple_channels=256, complex_channels=256,
            noise_mode='neuronal', noise_scale=0.35, noise_level=0.07, k_exc=25,
            model_arch='resnet50', image_size=224, visual_degrees=8, ksize=25, stride=4, cov_matrix = None,
            filters_r = None, filters_c = None):


    out_channels = simple_channels + complex_channels

    sf, theta, phase, nx, ny = generate_gabor_param(out_channels, gabor_seed, rand_param, sf_corr, sf_max, sf_min)

    gabor_params = {'simple_channels': simple_channels, 'complex_channels': complex_channels, 'rand_param': rand_param,
                    'gabor_seed': gabor_seed, 'sf_max': sf_max, 'sf_corr': sf_corr, 'sf': sf.copy(),
                    'theta': theta.copy(), 'phase': phase.copy(), 'nx': nx.copy(), 'ny': ny.copy()}
    arch_params = {'k_exc': k_exc, 'arch': model_arch, 'ksize': ksize, 'stride': stride}


    # Conversions
    ppd = image_size / visual_degrees

    sf = sf / ppd
    sigx = nx / sf
    sigy = ny / sf
    theta = theta/180 * np.pi
    phase = phase / 180 * np.pi

    vone_block = VOneBlockDN(sf=sf, theta=theta, sigx=sigx, sigy=sigy, phase=phase,
                           k_exc=k_exc, noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level,
                           simple_channels=simple_channels, complex_channels=complex_channels,
                           ksize=ksize, stride=stride, input_size=image_size, cov_matrix = cov_matrix,
                           filters_r = filters_r, filters_c = filters_c)

    if model_arch:
        bottleneck = nn.Conv2d(out_channels, 64, kernel_size=1, stride=1, bias=False)
        nn.init.kaiming_normal_(bottleneck.weight, mode='fan_out', nonlinearity='relu')

        if model_arch.lower() == 'resnet18':
            # following the same construction as under torchvision
            print("Model: ", "VOneResNet18")
            model_back_end = ResNetBackEnd(block=BasicBlock, layers=[2, 2, 2, 2])
        elif model_arch.lower() == 'resnet34':
            print('Model: ', 'VOneResnet34')
            model_back_end = ResNetBackEnd(block=BasicBlock, layers=[3, 4, 6, 3])
        elif model_arch.lower() == 'resnet50':
            print('Model: ', 'VOneResnet50')
            model_back_end = ResNetBackEnd(block=Bottleneck, layers=[3, 4, 6, 3])
        
        elif model_arch.lower() == 'alexnet':
            print('Model: ', 'VOneAlexNet')
            model_back_end = AlexNetBackEnd()
        elif model_arch.lower() == 'cornets':
            print('Model: ', 'VOneCORnet-S')
            model_back_end = CORnetSBackEnd()

        model = nn.Sequential(OrderedDict([
            ('vone_block', vone_block),
            ('bottleneck', bottleneck),
            ('model', model_back_end),
        ]))
    else:
        print('Model: ', 'VOneNet')
        model = vone_block

    model.image_size = image_size
    model.visual_degrees = visual_degrees
    model.gabor_params = gabor_params
    model.arch_params = arch_params

    return model

def downsampler(voneblock, ksize):

    ksz = (ksize,ksize)

    print(f"Interpolating to: {ksz}")
    
    voneblock.simple_conv_q0.weight = nn.Parameter(F.interpolate(voneblock.simple_conv_q0.weight, size = ksz), requires_grad = False)
    voneblock.simple_conv_q1.weight = nn.Parameter(F.interpolate(voneblock.simple_conv_q1.weight, size = ksz), requires_grad = False)

    voneblock.ksize = ksize

    voneblock.simple_conv_q0.padding = (voneblock.ksize//2, voneblock.ksize//2)
    voneblock.simple_conv_q1.padding = (voneblock.ksize//2, voneblock.ksize//2)

    return voneblock


def VOneNet(sf_corr=0.75, sf_max=9, sf_min=0, rand_param=False, gabor_seed=0,
            simple_channels=256, complex_channels=256,
            noise_mode='neuronal', noise_scale=0.35, noise_level=0.07, k_exc=25,
            model_arch='resnet50', image_size=224, visual_degrees=8, ksize=25, stride=4, use_TIN = False,
            rgb_seed = 0):



    out_channels = simple_channels + complex_channels

    sf, theta, phase, nx, ny = generate_gabor_param(out_channels, gabor_seed, rand_param, sf_corr, sf_max, sf_min)

    gabor_params = {'simple_channels': simple_channels, 'complex_channels': complex_channels, 'rand_param': rand_param,
                    'gabor_seed': gabor_seed, 'sf_max': sf_max, 'sf_corr': sf_corr, 'sf': sf.copy(),
                    'theta': theta.copy(), 'phase': phase.copy(), 'nx': nx.copy(), 'ny': ny.copy()}
    arch_params = {'k_exc': k_exc, 'arch': model_arch, 'ksize': ksize, 'stride': stride}


    # Conversions
    ppd = image_size / visual_degrees

    sf = sf / ppd
    sigx = nx / sf
    sigy = ny / sf
    theta = theta/180 * np.pi
    phase = phase / 180 * np.pi

    if use_TIN:
        ks = 25
    else:
        ks = ksize
    
    vone_block = VOneBlock(sf=sf, theta=theta, sigx=sigx, sigy=sigy, phase=phase,
                        k_exc=k_exc, noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level,
                        simple_channels=simple_channels, complex_channels=complex_channels,
                        ksize=ks, stride=stride, input_size=image_size, rgb_seed = rgb_seed)
        
    # vone_block = downsampler(vone_block, ksize = ksize)
    

    if model_arch:
        bottleneck = nn.Conv2d(out_channels, 64, kernel_size=1, stride=1, bias=False)
        nn.init.kaiming_normal_(bottleneck.weight, mode='fan_out', nonlinearity='relu')

        # particular form of initialisation that accelerates rate of convergence of cnn's during training

        if model_arch.lower() == 'resnet18':
            # following the same construction as under torchvision
            print("Model: ", "VOneResNet18")
            model_back_end = ResNetBackEnd(block=BasicBlock, layers=[2, 2, 2, 2])
        elif model_arch.lower() == 'resnet34':
            print('Model: ', 'VOneResnet34')
            model_back_end = ResNetBackEnd(block=BasicBlock, layers=[3, 4, 6, 3])
        elif model_arch.lower() == 'resnet50':
            print('Model: ', 'VOneResnet50')
            model_back_end = ResNetBackEnd(block=Bottleneck, layers=[3, 4, 6, 3])
        
        elif model_arch.lower() == 'alexnet':
            print('Model: ', 'VOneAlexNet')
            model_back_end = AlexNetBackEnd()
        elif model_arch.lower() == 'cornets':
            print('Model: ', 'VOneCORnet-S')
            model_back_end = CORnetSBackEnd()

        model = nn.Sequential(OrderedDict([
            ('vone_block', vone_block),
            ('bottleneck', bottleneck),
            ('model', model_back_end),
        ]))
    else:
        print('Model: ', 'VOneNet')
        model = vone_block

    model.image_size = image_size
    model.visual_degrees = visual_degrees
    model.gabor_params = gabor_params
    model.arch_params = arch_params

    return model

