#!/bin/bash
#shell script for full imagenet1k download.

python3 imagenet_download.py --output_path /datasets/imagenet --torrent_path torrents

cd /datasets/imagenet

mv ILSVRC2012_img_train.tar /datasets/imagenet/train/ && cd /datasets/imagenet/train && tar -xvf ILSVRC2012_img_train.tar
rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done

cd datasets/imagenet

mv ILSVRC2012_img_val.tar /datasets/imagenet/val && cd /datasets/imagenet/val && tar -xvf ILSVRC2012_img_val.tar
rm -f ILSVRC2012_img_test.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash