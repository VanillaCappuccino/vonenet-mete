#!/bin/bash
#shell script for full imagenet1k download.

python3 imagenet_p_download.py --output_path /datasets/imagenet-p

cd /datasets/imagenet-p

tar -xvf blur.tar
tar -xvf digital.tar
tar -xvf noise.tar
tar -xvf weather.tar

rm -f blur.tar
rm -f digital.tar
rm -f noise.tar
rm -f weather.tar