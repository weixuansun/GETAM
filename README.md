# GETAM

implementation for 'GETAM: Gradient-weighted Element-wise Transformer Attention Map for Weakly-supervised Semantic segmentation'

https://arxiv.org/abs/2112.02841

<img width="900" alt="image" src="https://user-images.githubusercontent.com/13931546/171977086-f043617e-422c-4a26-aa26-6904cf0416e7.png">

<img width="300" alt="image" src="https://user-images.githubusercontent.com/13931546/171977077-7fd509b3-a008-492c-aa51-88200bcd8b49.png">



## Step1: environment
- clone this repo 
```
git clone https://github.com/weixuansun/GETAM.git
```
- optionally create a new environment python>=3.6
- install requirements.txt
```
pip install -r requirements.txt
```
### optional: build python extension module for DenseEnergyLoss:
```
cd wrapper/bilateralfilter
python setup.py install
```
More details please see [here](https://github.com/meng-tang/rloss/tree/master/pytorch)


## Step2: dataset preparation
### pascal voc
- [Images](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) 
- [saliency map](https://1drv.ms/u/s!Ak3sXyXVg781yW78vNuDTblg2qdJ)

### MS-COCO 2014
- [Images](https://cocodataset.org/#home) 
- [saliency map](https://1drv.ms/u/s!Ak3sXyXVg781ymyj2mLjZJMTMF-G?e=1QWtp1)



## Step3: train and inference
### end-to-end training on pascal:
Training requires one GPU, you can change GPU setting accordingly.

    CUDA_VISIBLE_DEVICES=7 python train_from_init.py --session_name getam_001 -n 1 -g 1 -nr 0 --max_epoches 20  --lr 0.04 --cls_step 0.5 --seg_lr_scale 0.1 --sal_loss True --backbone vitb_hybrid --address 1234 --voc12_root {path to pascal voc dataset} --saliencypath {path to saliency maps}
    
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
  - Thanks for codebase provided by [RRM](https://github.com/zbf1991/RRM)

---
if you use this paper, please kindly cite:
```
@article{sun2021getam,
  title={GETAM: Gradient-weighted Element-wise Transformer Attention Map for Weakly-supervised Semantic segmentation},
  author={Sun, Weixuan and Zhang, Jing and Liu, Zheyuan and Zhong, Yiran and Barnes, Nick},
  journal={arXiv preprint arXiv:2112.02841},
  year={2021}
}
```

