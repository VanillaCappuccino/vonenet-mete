# #!/bin/bash

# python prepare_dataset.py
# PATH=$PATH:/scratch/mh2071/miniconda3/bin
# eval "$(conda shell.bash hook)"
# python train.py --in_path /scratch/mh2071/tiny-224 -o "/scratch/mh2071/checkpoints/vonenet-run1" --model_type vonenet --ngpus 2 --workers 2 --epochs 70 --save_model_epochs 1 train

sed -i -r "s/#user_allow_other/user_allow_other/ /etc/fuse.conf