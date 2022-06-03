import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from tool import imutils
import argparse
import cv2
import os.path
import torch.nn.functional as F
from DPT.DPT import DPTSegmentationModel
from myTool import *
from tool.metrics import Evaluator
from PIL import Image

classes = ['bkg',
    'aeroplane',
           'bicycle',
           'bird',
           'boat',
           'bottle',
           'bus',
           'car',
           'cat',
           'chair',
           'cow',
           'diningtable',
           'dog',
           'horse',
           'motorbike',
           'person',
           'pottedplant',
           'sheep',
           'sofa',
           'train',
           'tvmonitor']



def _crf_with_alpha(pred_prob, ori_img):
    bgcam_score = pred_prob.cpu().data.numpy()
    crf_score = imutils.crf_inference_inf(ori_img, bgcam_score,t=20, labels=21)

    return crf_score

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default='./netWeights/RRM_final.pth', type=str)
    parser.add_argument("--out_cam_pred", default='./output/result/no_crf', type=str)
    parser.add_argument("--out_la_crf", default='./output/result/crf', type=str)
    parser.add_argument("--out_color", default='./output/result/color', type=str)
    parser.add_argument("--LISTpath", default="./voc12/val_id.txt", type=str)
    parser.add_argument("--IMpath", default="/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages", type=str)
    parser.add_argument("--val", default=False, type=bool)
    parser.add_argument("--backbone", default="vitb_hybrid", type=str)

    args = parser.parse_args()

    model = DPTSegmentationModel(num_classes=20, backbone_name=args.backbone)
    weights_dict = torch.load(args.weights)
    model.load_state_dict(weights_dict, strict=False)

    model.eval()
    model.cuda()

    evaluator = Evaluator(num_class=21) 
    im_path = args.IMpath
    img_list = open(args.LISTpath).readlines()
    pred_softmax = torch.nn.Softmax(dim=0)

    if args.val == False:
        args.IMpath = '/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages2'
    else:
        args.IMpath = '/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages'
    im_path = args.IMpath
    print(args)


    for index, i in enumerate(img_list):
        print(index)

        if args.val == False:
            i = ((i.split('/'))[2])[0:-4]


        print(os.path.join(im_path, i[:-1] + '.jpg'))

        img_temp = cv2.imread(os.path.join(im_path, i[:-1] + '.jpg'))
        print(os.path.isfile(os.path.join(im_path, i[:-1] + '.jpg')))
        if args.val==True:
            target_path = os.path.join('/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/SegmentationClassAug', '{}.png'.format(i[:-1]))
            target = np.asarray(Image.open(target_path), dtype=np.int32)

        h, w, _ = img_temp.shape
        img_original = img_temp.astype(np.uint8)

        # test_size = 480

        # container = np.zeros((test_size, test_size, 3), np.float32)
        # if h>=w:
        #     img_temp = cv2.resize(img_temp, (int(test_size*w/h), test_size))
        #     # print(h,w, img_temp.shape)

        #     container[:, 0:int(test_size*w/h), :] = img_temp
        # else:
        #     img_temp = cv2.resize(img_temp, (test_size, int(test_size*h/w)))
        #     # print(h,w,img_temp.shape)
        #     container[0:int(test_size*h/w), :, :] = img_temp

        # img_temp = container

        output_list = []

        # multi-scale test
        for test_size in [256, 320, 480]:
            img_input = cv2.resize(img_temp, (test_size, test_size))

            img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB).astype(np.float)
            img_input[:, :, 0] = (img_input[:, :, 0] / 255. - 0.485) / 0.229
            img_input[:, :, 1] = (img_input[:, :, 1] / 255. - 0.456) / 0.224
            img_input[:, :, 2] = (img_input[:, :, 2] / 255. - 0.406) / 0.225

            input = torch.from_numpy(img_input[np.newaxis, :].transpose(0, 3, 1, 2)).float().cuda()

            _, output = model(input)

            # output = output[]
            # print(output.shape)

            # if h>=w:
            #     output = output[:,:,:,0:int(test_size*h/w)]
            # else:
            #     output = output[:,:,0:int(test_size*h/w), :]

            output = F.interpolate(output, (h, w),mode='bilinear',align_corners=False)
            output_list.append(output)

        output = (output_list[0] + output_list[1] + output_list[2])/3

        output = torch.squeeze(output)
        pred_prob = pred_softmax(output)

        output = torch.argmax(output,dim=0).cpu().numpy()
        save_path = os.path.join(args.out_cam_pred,i[:-1] + '.png')
        # cv2.imwrite(save_path,output.astype(np.uint8))
        
        if args.out_la_crf is not None:
            crf_la = _crf_with_alpha(pred_prob, img_original)

            crf_img = np.argmax(crf_la, 0)

            if args.val:
                evaluator.add_batch(target, crf_img)
                miou = evaluator.Mean_Intersection_over_Union()
                print(miou)

            # imageio.imsave(os.path.join(args.out_la_crf, i[:-1] + '.png'), crf_img.astype(np.uint8))

            # rgb_pred = decode_segmap(crf_img, dataset="pascal")
            # cv2.imwrite(os.path.join(args.out_color, i[:-1] + '_hybrid.png'),
            #             (rgb_pred * 255).astype('uint8') * 0.7 + img_original* 0.3)

            # if args.val:
            #     rgb_target = decode_segmap(target, dataset="pascal")
            #     cv2.imwrite(os.path.join(args.out_color, i[:-1] + '_gt.png'),
            #                 (rgb_target * 255).astype('uint8') * 0.7 + img_original* 0.3)

    if args.val:
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        print('Validation:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        for i in range(21):
            print(classes[i], evaluator.per_class_miou[i])
