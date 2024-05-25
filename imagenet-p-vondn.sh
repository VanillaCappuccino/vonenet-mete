#!/bin/bash

function runAllPerturbations(){
   arr=("$@")
   for i in "${arr[@]}";
      do
          python imagenet-p-eval.py -m vonenet -p "$i" --ngpu 1 --num_workers 16 \
          --cov_path 32x32x512x2525xl2norm \
          --vonenetdn_checkpoint checkpoints/vonenetdn-cov-variable-l2/latest_epoch.pth.tar \
          --data_dir /home/mete/repos/robustness/ImageNet-P/datasets/Tiny-ImageNet-P \
          --trainable_vonenetdn \
          --sf_min 0.5 --sf_max 11.2 \


          #-cp sthsth
      done

}

array=('gaussian_noise' 'shot_noise' 'motion_blur' 'zoom_blur' 'spatter' 'brightness' 'translate'
'rotate' 'tilt' 'scale' 'speckle_noise' 'gaussian_blur' 'snow' 'shear')

runAllPerturbations "${array[@]}"
