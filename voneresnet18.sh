# #!/bin/bash


python train_TIN.py --in_path tiny-imagenet-200 -o checkpoints/voneresnet18 --model_type vonenet --ngpus 1 --workers 16 --epochs 60 --save_model_epochs 1 train
