CUDA_VISIBLE_DEVICES=0 \
python train_from_init.py \
--session_name nitro_getam_001 \
-n 1 \
-g 1 \
-nr 0 \
--max_epoches 20  \
--lr 0.04 \
--cls_step 0.5 \
--seg_lr_scale 0.1 \
--sal_loss True \
--backbone vitb_hybrid \
--address 1234 \
--voc12_root /home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/ \
--saliencypath /home/users/u5876230/pascal_aug/pascal_saliency  \