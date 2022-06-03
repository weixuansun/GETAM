# GETAM

implementation for 'GETAM: Gradient-weighted Element-wise Transformer Attention Map for Weakly-supervised Semantic segmentation'

https://arxiv.org/abs/2112.02841

## Step1: environment
- create a new environment pytohn=3.6
- install requirements.txt


## Step2: dataset preparation



## Step3: train
    CUDA_VISIBLE_DEVICES=7 python train_from_init.py --session_name getam_001 -n 1 -g 1 -nr 0 --max_epoches 20  --lr 0.04 --cls_step 0.5 --seg_lr_scale 0.1 --sal_loss True --backbone vitb_hybrid --address 1234
