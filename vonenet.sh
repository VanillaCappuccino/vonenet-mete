# #!/bin/bash

# python prepare_dataset.py
# python train.py --in_path tiny-224 -o "../checkpoints/vonenet-run1" --model_type vonenetdn --ngpus 2 --workers 2 --epochs 5 train

sed -i -r "s/#user_allow_other/user_allow_other/ /etc/fuse.conf