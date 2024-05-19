# #!/bin/bash


python train_TIN.py --in_path tiny-imagenet-200 -o checkpoints/resnet-replicate --model_type barebones --ngpus 1 --workers 2 --epochs 60 --save_model_epochs 1 train