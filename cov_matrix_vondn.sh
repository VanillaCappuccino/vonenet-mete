python cov_matrix_computer.py --in_path "" --out-path 32x32x128x2525 \
python train_TIN.py --in_path tiny-imagenet-200 --cov_path 32x32x128x2525 -o checkpoints/vonenetdn-cov-128-2525 --model_type vonenetdn --ngpus 1 --workers 16 --epochs 60 --save_model_epochs 1 --latest_only train
