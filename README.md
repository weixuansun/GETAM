# GETAM

implementation for 'GETAM: Gradient-weighted Element-wise Transformer Attention Map for Weakly-supervised Semantic segmentation'

https://arxiv.org/abs/2112.02841

## Step1: environment
- create a new environment pytohn=3.6
- install requirements.txt


## Step2: dataset preparation
### pascal voc
- [Images](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) 
- [saliency map](https://1drv.ms/u/s!Ak3sXyXVg781yW78vNuDTblg2qdJ)

### MS-COCO 2014
- [Images](https://cocodataset.org/#home) 
- saliency map



## Step3: train and inference
### end-to-end training on pascal:

    CUDA_VISIBLE_DEVICES=7 python train_from_init.py --session_name getam_001 -n 1 -g 1 -nr 0 --max_epoches 20  --lr 0.04 --cls_step 0.5 --seg_lr_scale 0.1 --sal_loss True --backbone vitb_hybrid --address 1234

### Inference on pascal
    python test.py --weights {path to weight} --val True --backbone {backbone}
    
### end-to-end training on COCO
coming soon

## Checkpoints

  | Dataset         | Backbone | mIoU(val) | mIoU(test) |Checkpoint           |
  | --------------- | ------ | -----------  |----|---                 |
  | PASCAL VOC 2012 | Vit_hybrid   | 71.7  |  72.3   | [Download](https://drive.google.com/file/d/1gOtbWFpi2OTn7NPIPOomnn6cfELgVVVA/view?usp=sharing)      | 
  | PASCAL VOC 2012 | Vit   | 68.1       | 68.8 |[Download](https://drive.google.com/file/d/16jMDJTdiXnpax0uhNMgwgPLU6INAcGA4/view?usp=sharing)      |
  | PASCAL VOC 2012 | deit   |66.0       | 68.9 |[Download](https://drive.google.com/file/d/1XS1YnIjB64EMGPUMOWAl9f1OqIxgVw2Y/view?usp=sharing)      |
  | PASCAL VOC 2012 | deit_distilled   | 70.7  | 71.1    | [Download](https://drive.google.com/file/d/1pFs0ZSAoIwxa1K4P5xxNJSK1-g_5MwkH/view?usp=sharing)      |
  | COCO | vit_hybrid   | 36.4  |     | [Download](https://drive.google.com/file/d/1LvswI1_B76yalEdUSv2QhxTxCU120y1h/view?usp=sharing)      |
  
  
  ## Acknowledgement
  - Thanks for the saliency object detection code provided by [UCNet](https://github.com/JingZhang617/UCNet)
  - Thanks for codebase provided by [DPT](https://github.com/isl-org/DPT)
