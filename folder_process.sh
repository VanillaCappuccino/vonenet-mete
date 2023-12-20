#!/bin/bash

python3 torrent_download.py

mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train && tar -xvf ILSVRC2012_img_train.tar
rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
rm -f ILSVRC2012_img_test.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
cd ..

python3 setup.py install
python3 run.py "" --model_arch "alexnet"