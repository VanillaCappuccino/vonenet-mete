#!/bin/bash
python imagenet-c-eval.py -m resnet18 --ngpu 1 --rn18_checkpoint checkpoints/rn18_non_pretrained/epoch_60.pth.tar --data_dir /home/mete/repos/robustness/ImageNet-C/datasets/Tiny-ImageNet-C