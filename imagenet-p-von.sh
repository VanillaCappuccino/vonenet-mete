#!/bin/bash

function runAllPerturbations(){
   arr=("$@")
   for i in "${arr[@]}";
      do
          python imagenet-p-eval.py -m vonenet --identifier fixed -p "$i" --ngpu 1 --num_workers 16 --vonenet_checkpoint checkpoints/voneresnet18-noisy-fixed/epoch_60.pth.tar --data_dir /home/mete/repos/robustness/ImageNet-P/datasets/Tiny-ImageNet-P
          #-cp sthsth
      done

}

array=('gaussian_noise' 'shot_noise' 'motion_blur' 'zoom_blur' 'spatter' 'brightness' 'translate'
'rotate' 'tilt' 'scale' 'speckle_noise' 'gaussian_blur' 'snow' 'shear')

runAllPerturbations "${array[@]}"
