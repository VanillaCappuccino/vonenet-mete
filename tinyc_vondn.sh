#!/bin/bash
python imagenet-c-eval.py -m resnet18 --identifier vonenetdn \
--cov_path 32x32x512x2525xl2norm \
--trainable_vonenetdn --ngpu 1\
--sf_min 0.5 --sf_max 11.2 \
--vonenetdn_checkpoint checkpoints/vonenetdn-cov-variable-l2/latest_epoch.pth.tar \
--data_dir /home/mete/repos/robustness/ImageNet-C/datasets/Tiny-ImageNet-C \