# #!/bin/bash


python train_TIN.py --in_path tiny-imagenet-200 -o checkpoints/rn18_in1k --model_type barebones --ngpus 1 --workers 16 --epochs 60 --imagenet1k_ckpt --save_model_epochs 1 train
