#!/bin/bash
python imagenet-c-eval.py -m resnet18 --identifier voneresnet18-noisy-fixed --ngpu 1 --vonenet_checkpoint checkpoints/voneresnet18-noisy-fixed/epoch_60.pth.tar --data_dir /home/mete/repos/robustness/ImageNet-C/datasets/Tiny-ImageNet-C