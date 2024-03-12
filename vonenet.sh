#!/bin/bash

python prepare_dataset.py
python train.py --in_path tiny-224 -o vonenet-run1 --model_type vonenet --ngpus 2 --workers 4 --epochs 5 train