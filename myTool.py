import numpy as np
from numpy.lib.function_base import piecewise
from torch.functional import norm
import tool.imutils as imutils
import torch
import torch.nn.functional as F
import cv2
import random
import os
import matplotlib.pyplot as plt
import PIL.Image
import math
    

classes = ['aeroplane',
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

def pred_acc(original, predicted):
    label_count = int(torch.sum(original))
    ind = np.argpartition(predicted, -label_count)[0][-label_count:]

    predicted_binary = torch.zeros_like(original)
    predicted_binary[ind] = 1
    return predicted_binary.eq(original).sum().numpy()/len(original)

def _crf_with_alpha(ori_img,cam_dict, alpha):
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = imutils.crf_inference(ori_img, bgcam_score, labels=bgcam_score.shape[0])

    n_crf_al = np.zeros([21, bg_score.shape[1], bg_score.shape[2]])
    n_crf_al[0, :, :] = crf_score[0, :, :]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key + 1] = crf_score[i + 1]

    return n_crf_al



# use this 
def compute_seg_label_3(ori_img, cam_label, norm_cam, name, iter, saliency, cls_pred, save_heatmap=False, cut_threshold = 0.9):
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    cam_label = cam_label.astype(np.uint8)

    cam_dict = {}
    cam_np = np.zeros_like(norm_cam)
    for i in range(20):
        if cam_label[i] > 1e-5:
            cam_dict[i] = norm_cam[i]
            cam_np[i] = norm_cam[i]
    
    # save heatmap
    if save_heatmap:
        img = ori_img
        keys = list(cam_dict.keys())
        for target_class in keys:
            mask = cam_dict[target_class]
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))
            cam_output = heatmap * 0.5 + img * 0.5
            cv2.imwrite(os.path.join('/home/users/u5876230/ete_project/ete_output/vis/', name + '_{}_heatmap_orig.jpg'.format(classes[target_class])), cam_output)

    _, h, w = norm_cam.shape
    
    bg_score = np.power(1 - np.max(cam_np, 0), 32)
    bg_score = np.expand_dims(bg_score, axis=0)
    cam_all = np.concatenate((bg_score, cam_np))

    bkg_high_conf_area = np.zeros([h, w], dtype=bool)

    crf_label = np.argmax(cam_all, 0)

    crf_label[crf_label == 0 ] = 255
    crf_label[saliency == 0 ] = 0

    for class_i in range(20):
        if cam_label[class_i] > 1e-5:
            cam_class = norm_cam[class_i, :,:]
            cam_class_order = cam_class[cam_class > 0]
            cam_class_order = np.sort(cam_class_order)
            confidence_pos = int(cam_class_order.shape[0] * cut_threshold)
            if confidence_pos>0:
                confidence_value = cam_class_order[confidence_pos]
                bkg_high_conf_cls = np.logical_and((cam_class>confidence_value), (crf_label==0))
                crf_label[bkg_high_conf_cls] = class_i+1
                saliency[bkg_high_conf_cls] = 255
                bkg_high_conf_conflict = np.logical_and(bkg_high_conf_cls, bkg_high_conf_area)
                crf_label[bkg_high_conf_conflict] = 255

                bkg_high_conf_area[bkg_high_conf_cls] = 1

    # remove background noise
    frg = ((crf_label != 0) *  255).astype('uint8')
    frg_dilate = cv2.morphologyEx(frg, cv2.MORPH_OPEN, kernel=np.ones((10,10),np.uint8 ))
    crf_label[frg_dilate!=255] = 0

    # cv2.imwrite('/data/u5876230/ete_wsss/pseudo/{}.png'.format(name), crf_label)

    # rgb_pseudo_label = decode_segmap(crf_label, dataset="pascal")
    # cv2.imwrite('/home/users/u5876230/ete_project/ete_output/pseudo/{}_color.png'.format(name),
                        # (rgb_pseudo_label * 255).astype('uint8') * 0.7 + ori_img * 0.3)
    # cv2.imwrite('/home/users/u5876230/ete_project/ete_output/vis/{}_orig.png'.format(name),ori_img)

    return crf_label, saliency

# use this for coco
def compute_seg_label_coco(ori_img, cam_label, norm_cam, croppings, name, iter, saliency, cls_pred, save_heatmap=False, cut_threshold = 0.3):
    cam_label = cam_label.astype(np.uint8)

    cam_dict = {}
    cam_np = np.zeros_like(norm_cam)
    for i in range(80):
        if cam_label[i] > 1e-5:
            cam_dict[i] = norm_cam[i]
            cam_np[i] = norm_cam[i]
    
    # save heatmap
    if save_heatmap:
        img = ori_img
        keys = list(cam_dict.keys())
        for target_class in keys:
            mask = cam_dict[target_class]
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))
            cam_output = heatmap * 0.5 + img * 0.5
            cv2.imwrite(os.path.join('/home/users/u5876230/ete_project/ete_output/heatmap/', name + '_{}.jpg'.format(classes[target_class])), cam_output)

    output = cls_pred.detach().cpu()
    accuracy = pred_acc(torch.from_numpy(cam_label), output.unsqueeze(0))
    # if accuracy < 1:
    #    return np.ones((norm_cam.shape[1], norm_cam.shape[1])) * 255

    _, h, w = norm_cam.shape
    
    # if np.sum(cam_label)<2: # one class simple image
    bg_score = np.power(1 - np.max(cam_np, 0), 32)
    bg_score = np.expand_dims(bg_score, axis=0)
    cam_all = np.concatenate((bg_score, cam_np))

    bkg_high_conf_area = np.zeros([h, w], dtype=bool)

    crf_label = np.argmax(cam_all, 0)
    crf_label[crf_label ==0 ] = 255
    crf_label[saliency == 0 ] = 0
    for class_i in range(80):
        if cam_label[class_i] > 1e-5:
            cam_class = norm_cam[class_i, :,:]
            cam_class_order = cam_class[cam_class > 0]
            cam_class_order = np.sort(cam_class_order)
            confidence_pos = int(cam_class_order.shape[0] * 0.95)
            if confidence_pos>0:
                confidence_value = cam_class_order[confidence_pos]

                bkg_high_conf_cls = np.logical_and((cam_class>confidence_value), (crf_label==0))
                crf_label[bkg_high_conf_cls] = class_i+1
                saliency[bkg_high_conf_cls] = 255
                bkg_high_conf_conflict = np.logical_and(bkg_high_conf_cls, bkg_high_conf_area)
                crf_label[bkg_high_conf_conflict] = 255

                bkg_high_conf_area[bkg_high_conf_cls] = 1
    
    # remove background noise
    frg = ((crf_label != 0) *  255).astype('uint8')
    frg_dilate = cv2.morphologyEx(frg, cv2.MORPH_OPEN, kernel=np.ones((10,10),np.uint8 ))
    crf_label[frg_dilate!=255] = 0
    
    cv2.imwrite('/home/users/u5876230/ete_project/pseudo_label/{}.png'.format(name), crf_label)

    rgb_pseudo_label = decode_segmap(crf_label, dataset="pascal")
    cv2.imwrite('/home/users/u5876230/ete_project/ete_output/pseudo/{}_color.png'.format(name),
                        (rgb_pseudo_label * 255).astype('uint8') * 0.5 + ori_img * 0.5)

    # cv2.imwrite('/home/users/u5876230/ete_project/ete_output/pseudo/{}.png'.format(name),
    #                     (saliency).astype('uint8'))

    # cv2.imwrite('/home/users/u5876230/ete_project/ete_output/saliency_pseudo/{}.png'.format(name),
    #                     (saliency).astype('uint8'))
    
    return crf_label, saliency



def compute_joint_loss(ori_img, seg, seg_label, croppings, critersion, DenseEnergyLosslayer):
    seg_label = np.expand_dims(seg_label,axis=1)
    seg_label = torch.from_numpy(seg_label)

    w = seg_label.shape[2]
    h = seg_label.shape[3]
    pred = F.interpolate(seg,(w,h),mode="bilinear",align_corners=False)
    pred_softmax = torch.nn.Softmax(dim=1)
    pred_probs = pred_softmax(pred)
    ori_img = torch.from_numpy(ori_img.astype(np.float32))
    croppings = torch.from_numpy(croppings.astype(np.float32).transpose(2,0,1))
    dloss = DenseEnergyLosslayer(ori_img,pred_probs,croppings, seg_label)
    dloss = dloss.cuda()

    seg_label_tensor = seg_label.long().cuda()

    seg_label_copy = seg_label_tensor.clone().squeeze(1)
    # seg_label_copy = torch.squeeze(seg_label_tensor.clone())
 
    # print(seg_label_copy.shape)
    bg_label = seg_label_copy.clone()
    fg_label = seg_label_copy.clone()
    bg_label[seg_label_copy != 0] = 255
    fg_label[seg_label_copy == 0] = 255

    # print(pred.shape, bg_label.shape)
    bg_celoss = critersion(pred, bg_label.long().cuda())

    fg_celoss = critersion(pred, fg_label.long().cuda())

    celoss = bg_celoss + fg_celoss

    return celoss, dloss


def compute_cam_up(cam, label, w, h, b):
    cam_up = F.interpolate(cam, (w, h), mode='bilinear', align_corners=False)
    cam_up = cam_up * label.clone().view(b, 20, 1, 1)
    cam_up = cam_up.cpu().data.numpy()
    return cam_up


def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
            # img_list.append(line[12:23])
    return img_list

def read_file_2(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[12:23])
    return img_list

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def resize_label_batch(label, size):
    label_resized = np.zeros((size, size, 1, label.shape[3]))
    interp = torch.nn.UpsamplingBilinear2d(size=(size, size))
    labelVar = torch.autograd.Variable(torch.from_numpy(label.transpose(3, 2, 0, 1)))
    label_resized[:, :, :, :] = interp(labelVar).data.numpy().transpose(2, 3, 1, 0)
    label_resized[label_resized>21] = 255
    return label_resized


def flip(I, flip_p):
    if flip_p > 0.5:
        return np.fliplr(I)
    else:
        return I

def flip2(I,saliency, flip_p):
    if flip_p > 0.5:
        return np.fliplr(I), np.fliplr(saliency)
    else:
        return I, saliency

def scale_im(img_temp, scale):
    new_dims = (int(img_temp.shape[1] * scale), int(img_temp.shape[0] * scale))
    return cv2.resize(img_temp, new_dims).astype(float)


def scale_gt(img_temp, scale):
    new_dims = (int(img_temp.shape[1] * scale), int(img_temp.shape[0] * scale))
    return cv2.resize(img_temp, new_dims, interpolation=cv2.INTER_NEAREST).astype(float)

def load_image_label_list_from_npy(img_name_list):

    cls_labels_dict = np.load('voc12/cls_labels.npy',allow_pickle=True).item()

    return [cls_labels_dict[img_name] for img_name in img_name_list]


def RandomCrop(imgarr, cropsize):

    h, w, c = imgarr.shape

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space+1)
    else:
        cont_left = random.randrange(-w_space+1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space+1)
    else:
        cont_top = random.randrange(-h_space+1)
        img_top = 0

    img_container = np.zeros((cropsize, cropsize, imgarr.shape[-1]), np.float32)

    cropping =  np.zeros((cropsize, cropsize), np.bool)

    img_container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        imgarr[img_top:img_top+ch, img_left:img_left+cw]
    cropping[cont_top:cont_top + ch, cont_left:cont_left + cw] = 1

    return img_container, cropping

def RandomCrop2(imgarr, saliency, cropsize):

    h, w, c = imgarr.shape

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space+1)
    else:
        cont_left = random.randrange(-w_space+1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space+1)
    else:
        cont_top = random.randrange(-h_space+1)
        img_top = 0

    img_container = np.zeros((cropsize, cropsize, imgarr.shape[-1]), np.float32)
    saliency_container = np.zeros((cropsize, cropsize), np.float32)

    cropping =  np.zeros((cropsize, cropsize), np.bool)

    img_container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        imgarr[img_top:img_top+ch, img_left:img_left+cw]
    saliency_container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        saliency[img_top:img_top+ch, img_left:img_left+cw]
    
    cropping[cont_top:cont_top + ch, cont_left:cont_left + cw] = 1

    return img_container, cropping, saliency_container

def RandomResizeLong(img, min_long, max_long):
    target_long = random.randint(min_long, max_long)
    h,w, c = img.shape

    if w < h:
        target_shape = (int(round(w * target_long / h)), target_long)
    else:
        target_shape = (target_long, int(round(h * target_long / w)))

    # img = img.resize(target_shape, resample=PIL.Image.CUBIC)
    img = cv2.resize(img, target_shape)

    return img

def RandomResizeLong2(img,saliency, min_long, max_long):
    target_long = random.randint(min_long, max_long)
    h,w, c = img.shape

    if w < h:
        target_shape = (int(round(w * target_long / h)), target_long)
    else:
        target_shape = (target_long, int(round(h * target_long / w)))

    # img = img.resize(target_shape, resample=PIL.Image.CUBIC)
    img = cv2.resize(img, target_shape)
    saliency = cv2.resize(saliency, target_shape, interpolation=cv2.INTER_NEAREST)

    return img, saliency

def CenterCrop(npimg, cropsize, default_value = 0):
    h, w = npimg.shape[:2]

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    sh = h - cropsize
    sw = w - cropsize

    if sw > 0:
        cont_left = 0
        img_left = int(round(sw / 2))
    else:
        cont_left = int(round(-sw / 2))
        img_left = 0

    if sh > 0:
        cont_top = 0
        img_top = int(round(sh / 2))
    else:
        cont_top = int(round(-sh / 2))
        img_top = 0

    if len(npimg.shape) == 2:
        container = np.ones((cropsize, cropsize), npimg.dtype)*default_value
    else:
        container = np.ones((cropsize, cropsize, npimg.shape[2]), npimg.dtype)*default_value
    cropping =  np.zeros((cropsize, cropsize), np.bool)

    container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        npimg[img_top:img_top+ch, img_left:img_left+cw]
    cropping[cont_top:cont_top + ch, cont_left:cont_left + cw] = 1


    return container, cropping

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability # Random Erasing probability
        self.mean = mean
        self.sl = sl
        self.sh = sh # max erasing area
        self.r1 = r1 # aspect of erasing area
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

def hide_patch(img):
    # get width and height of the image
    s = img.shape
    wd = s[0]
    ht = s[1]

    # possible grid size, 0 means no hiding
    grid_sizes=[0,16,32,44,56]

    # hiding probability
    hide_prob = 0.5
 
    # randomly choose one grid size
    grid_size= grid_sizes[random.randint(0,len(grid_sizes)-1)]

    # hide the patches
    if(grid_size>0):
         for x in range(0,wd,grid_size):
             for y in range(0,ht,grid_size):
                 x_end = min(wd, x+grid_size)  
                 y_end = min(ht, y+grid_size)
                 if(random.random() <=  hide_prob):
                       img[x:x_end,y:y_end,:]=0

    return img

def get_data_from_chunk_v2(chunk, args):
    img_path = args.IMpath

    scale = np.random.uniform(0.7, 1.3)
    dim = args.crop_size
    images = np.zeros((dim, dim, 3, len(chunk)))
    ori_images = np.zeros((dim, dim, 3, len(chunk)),dtype=np.uint8)
    croppings = np.zeros((dim, dim, len(chunk)))
    labels = load_image_label_list_from_npy(chunk)
    labels = torch.from_numpy(np.array(labels))

    name_list = []

    for i, piece in enumerate(chunk):
        name_list.append(piece)
        flip_p = np.random.uniform(0, 1)
        img_temp = cv2.imread(os.path.join(img_path, piece + '.jpg'))
        img_temp = cv2.cvtColor(img_temp,cv2.COLOR_BGR2RGB).astype(np.float)
        # img_temp = scale_im(img_temp, scale)
        img_temp = RandomResizeLong(img_temp, int(dim*0.9), int(dim/0.875))
        img_temp = flip(img_temp, flip_p)
        img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
        img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
        img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225
        img_temp, cropping = RandomCrop(img_temp, dim)
        # img_temp = hide_patch(img_temp)
        ori_temp = np.zeros_like(img_temp)
        ori_temp[:, :, 0] = (img_temp[:, :, 0] * 0.229 + 0.485) * 255.
        ori_temp[:, :, 1] = (img_temp[:, :, 1] * 0.224 + 0.456) * 255.
        ori_temp[:, :, 2] = (img_temp[:, :, 2] * 0.225 + 0.406) * 255.
        ori_images[:, :, :, i] = ori_temp.astype(np.uint8)
        croppings[:,:,i] = cropping.astype(np.float32)

        images[:, :, :, i] = img_temp

    images = images.transpose((3, 2, 0, 1))
    ori_images = ori_images.transpose((3, 2, 0, 1))
    images = torch.from_numpy(images).float()
    return images, ori_images, labels, croppings, name_list


def get_data_from_chunk_v3(chunk, args):
    img_path = args.IMpath

    scale = np.random.uniform(0.7, 1.3)

    dim = args.crop_size
    images = np.zeros((dim, dim, 3, len(chunk)))
    saliency = np.zeros((dim, dim, len(chunk)))

    ori_images = np.zeros((dim, dim, 3, len(chunk)),dtype=np.uint8)
    croppings = np.zeros((dim, dim, len(chunk)))
    labels = load_image_label_list_from_npy(chunk)
    labels = torch.from_numpy(np.array(labels))

    name_list = []

    for i, piece in enumerate(chunk):
        name_list.append(piece)
        flip_p = np.random.uniform(0, 1)
        img_temp = cv2.imread(os.path.join(img_path, piece + '.jpg'))
        img_temp = cv2.cvtColor(img_temp,cv2.COLOR_BGR2RGB).astype(np.float)
        saliency_map_path = os.path.join(args.saliencypath, '{}.png'.format(piece))
        saliency_map = PIL.Image.open(saliency_map_path)
        saliency_map = np.asarray(saliency_map)
        # img_temp = scale_im(img_temp, scale)
        img_temp, saliency_map = RandomResizeLong2(img_temp, saliency_map, int(dim*0.9), int(dim/0.875))

        img_temp, saliency_map = flip2(img_temp,saliency_map, flip_p)

        img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
        img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
        img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225
        img_temp, cropping, saliency_map = RandomCrop2(img_temp,saliency_map, dim)

        ori_temp = np.zeros_like(img_temp)
        ori_temp[:, :, 0] = (img_temp[:, :, 0] * 0.229 + 0.485) * 255.
        ori_temp[:, :, 1] = (img_temp[:, :, 1] * 0.224 + 0.456) * 255.
        ori_temp[:, :, 2] = (img_temp[:, :, 2] * 0.225 + 0.406) * 255.
        ori_images[:, :, :, i] = ori_temp.astype(np.uint8)
        croppings[:,:,i] = cropping.astype(np.float32)

        images[:, :, :, i] = img_temp
        saliency[:, :,  i] = saliency_map

    images = images.transpose((3, 2, 0, 1))
    saliency = saliency.transpose((2, 0, 1))
    ori_images = ori_images.transpose((3, 2, 0, 1))
    images = torch.from_numpy(images).float()
    saliency = torch.from_numpy(saliency).float()

    return images, ori_images, labels, croppings, name_list, saliency


# get both target map and image for validation
def get_data_from_chunk_v4(chunk, args):
    img_path = args.IMpath

    scale = np.random.uniform(0.7, 1.3)

    dim = args.crop_size
    images = np.zeros((dim, dim, 3, len(chunk)))
    target = np.zeros((dim, dim, len(chunk)))

    ori_images = np.zeros((dim, dim, 3, len(chunk)),dtype=np.uint8)
    croppings = np.zeros((dim, dim, len(chunk)))
    labels = load_image_label_list_from_npy(chunk)
    labels = torch.from_numpy(np.array(labels))

    name_list = []

    for i, piece in enumerate(chunk):
        name_list.append(piece)
        flip_p = np.random.uniform(0, 1)
        img_temp = cv2.imread(os.path.join(img_path, piece + '.jpg'))
        img_temp = cv2.cvtColor(img_temp,cv2.COLOR_BGR2RGB).astype(np.float)
        target_path = os.path.join('/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/SegmentationClassAug/', '{}.png'.format(piece))
        target_map = PIL.Image.open(target_path)
        target_map = np.asarray(target_map)
        # print(target_map.shape)
        # img_temp = scale_im(img_temp, scale)
        # target_map = scale_im(target_map, scale)
        img_temp, target_map = RandomResizeLong2(img_temp, target_map, int(dim), int(dim))

        img_temp, target_map = flip2(img_temp,target_map, flip_p)

        img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
        img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
        img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225
        img_temp, cropping, target_map = RandomCrop2(img_temp,target_map, dim)

        ori_temp = np.zeros_like(img_temp)
        ori_temp[:, :, 0] = (img_temp[:, :, 0] * 0.229 + 0.485) * 255.
        ori_temp[:, :, 1] = (img_temp[:, :, 1] * 0.224 + 0.456) * 255.
        ori_temp[:, :, 2] = (img_temp[:, :, 2] * 0.225 + 0.406) * 255.
        ori_images[:, :, :, i] = ori_temp.astype(np.uint8)
        croppings[:,:,i] = cropping.astype(np.float32)

        # print(np.unique(target_map))
        images[:, :, :, i] = img_temp
        target[:, :,  i] = target_map

    images = images.transpose((3, 2, 0, 1))
    target = target.transpose((2, 0, 1))
    ori_images = ori_images.transpose((3, 2, 0, 1))
    images = torch.from_numpy(images).float()
    target = torch.from_numpy(target).float()

    return images, ori_images, labels, croppings, name_list, target


def get_data_from_chunk_v5(chunk, args):
    img_path = args.IMpath

    scale = np.random.uniform(0.7, 1.3)

    dim = args.crop_size
    images = np.zeros((dim, dim, 3, len(chunk)))
    saliency = np.zeros((dim, dim, len(chunk)))

    ori_images = np.zeros((dim, dim, 3, len(chunk)),dtype=np.uint8)
    croppings = np.zeros((dim, dim, len(chunk)))
    labels = load_image_label_list_from_npy(chunk)
    labels = torch.from_numpy(np.array(labels))

    name_list = []

    for i, piece in enumerate(chunk):
        name_list.append(piece)
        flip_p = np.random.uniform(0, 1)
        img_temp = cv2.imread(os.path.join(img_path, piece + '.jpg'))
        img_temp = cv2.cvtColor(img_temp,cv2.COLOR_BGR2RGB).astype(np.float)
        saliency_map_path = os.path.join('/home/users/u5876230/swin_sod/pascal/', '{}.png'.format(piece))
        saliency_map = PIL.Image.open(saliency_map_path)
        saliency_map = np.asarray(saliency_map)
       
        img_temp =  cv2.resize(img_temp, (256,256))
        saliency_map =  cv2.resize(saliency_map, (256,256))

        img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
        img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
        img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225

        ori_temp = np.zeros_like(img_temp)
        ori_temp[:, :, 0] = (img_temp[:, :, 0] * 0.229 + 0.485) * 255.
        ori_temp[:, :, 1] = (img_temp[:, :, 1] * 0.224 + 0.456) * 255.
        ori_temp[:, :, 2] = (img_temp[:, :, 2] * 0.225 + 0.406) * 255.
        ori_images[:, :, :, i] = ori_temp.astype(np.uint8)
        croppings[:,:,i] = None

        images[:, :, :, i] = img_temp
        saliency[:, :,  i] = saliency_map

    images = images.transpose((3, 2, 0, 1))
    saliency = saliency.transpose((2, 0, 1))
    ori_images = ori_images.transpose((3, 2, 0, 1))
    images = torch.from_numpy(images).float()
    saliency = torch.from_numpy(saliency).float()

    return images, ori_images, labels, croppings, name_list, saliency


def get_data_from_chunk_val(chunk, args):
    img_path = args.IMpath

    scale = np.random.uniform(0.7, 1.3)
    dim = args.crop_size
    images = np.zeros((dim, dim, 3, len(chunk)))
    ori_images = np.zeros((dim, dim, 3, len(chunk)),dtype=np.uint8)
    croppings = np.zeros((dim, dim, len(chunk)))
    labels = load_image_label_list_from_npy(chunk)
    labels = torch.from_numpy(np.array(labels))

    name_list = []

    for i, piece in enumerate(chunk):
        name_list.append(piece)
        flip_p = np.random.uniform(0, 1)
        img_temp = cv2.imread(os.path.join(img_path, piece + '.jpg'))
        img_temp = cv2.cvtColor(img_temp,cv2.COLOR_BGR2RGB).astype(np.float)
        # img_temp = scale_im(img_temp, scale)

        img_temp = RandomResizeLong(img_temp, int(dim), int(dim/0.875))
        # img_temp =  cv2.resize(img_temp, (256,256))

        img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
        img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
        img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225
        img_temp, cropping = CenterCrop(img_temp, dim)

        ori_temp = np.zeros_like(img_temp)
        ori_temp[:, :, 0] = (img_temp[:, :, 0] * 0.229 + 0.485) * 255.
        ori_temp[:, :, 1] = (img_temp[:, :, 1] * 0.224 + 0.456) * 255.
        ori_temp[:, :, 2] = (img_temp[:, :, 2] * 0.225 + 0.406) * 255.
        ori_images[:, :, :, i] = ori_temp.astype(np.uint8)
        # croppings[:,:,i] = cropping.astype(np.float32)

        images[:, :, :, i] = img_temp

    images = images.transpose((3, 2, 0, 1))
    ori_images = ori_images.transpose((3, 2, 0, 1))
    images = torch.from_numpy(images).float()
    return images, ori_images, labels, name_list



classes = [{"supercategory": "person", "id": 1, "name": "person"}, # 一共80类
               {"supercategory": "vehicle", "id": 2, "name": "bicycle"},
               {"supercategory": "vehicle", "id": 3, "name": "car"},
               {"supercategory": "vehicle", "id": 4, "name": "motorcycle"},
               {"supercategory": "vehicle", "id": 5, "name": "airplane"},
               {"supercategory": "vehicle", "id": 6, "name": "bus"},
               {"supercategory": "vehicle", "id": 7, "name": "train"},
               {"supercategory": "vehicle", "id": 8, "name": "truck"},
               {"supercategory": "vehicle", "id": 9, "name": "boat"},
               {"supercategory": "outdoor", "id": 10, "name": "traffic light"},
               {"supercategory": "outdoor", "id": 11, "name": "fire hydrant"},
               {"supercategory": "outdoor", "id": 13, "name": "stop sign"},
               {"supercategory": "outdoor", "id": 14, "name": "parking meter"},
               {"supercategory": "outdoor", "id": 15, "name": "bench"},
               {"supercategory": "animal", "id": 16, "name": "bird"},
               {"supercategory": "animal", "id": 17, "name": "cat"},
               {"supercategory": "animal", "id": 18, "name": "dog"},
               {"supercategory": "animal", "id": 19, "name": "horse"},
               {"supercategory": "animal", "id": 20, "name": "sheep"},
               {"supercategory": "animal", "id": 21, "name": "cow"},
               {"supercategory": "animal", "id": 22, "name": "elephant"},
               {"supercategory": "animal", "id": 23, "name": "bear"},
               {"supercategory": "animal", "id": 24, "name": "zebra"},
               {"supercategory": "animal", "id": 25, "name": "giraffe"},
               {"supercategory": "accessory", "id": 27, "name": "backpack"},
               {"supercategory": "accessory", "id": 28, "name": "umbrella"},
               {"supercategory": "accessory", "id": 31, "name": "handbag"},
               {"supercategory": "accessory", "id": 32, "name": "tie"},
               {"supercategory": "accessory", "id": 33, "name": "suitcase"},
               {"supercategory": "sports", "id": 34, "name": "frisbee"},
               {"supercategory": "sports", "id": 35, "name": "skis"},
               {"supercategory": "sports", "id": 36, "name": "snowboard"},
               {"supercategory": "sports", "id": 37, "name": "sports ball"},
               {"supercategory": "sports", "id": 38, "name": "kite"},
               {"supercategory": "sports", "id": 39, "name": "baseball bat"},
               {"supercategory": "sports", "id": 40, "name": "baseball glove"},
               {"supercategory": "sports", "id": 41, "name": "skateboard"},
               {"supercategory": "sports", "id": 42, "name": "surfboard"},
               {"supercategory": "sports", "id": 43, "name": "tennis racket"},
               {"supercategory": "kitchen", "id": 44, "name": "bottle"},
               {"supercategory": "kitchen", "id": 46, "name": "wine glass"},
               {"supercategory": "kitchen", "id": 47, "name": "cup"},
               {"supercategory": "kitchen", "id": 48, "name": "fork"},
               {"supercategory": "kitchen", "id": 49, "name": "knife"},
               {"supercategory": "kitchen", "id": 50, "name": "spoon"},
               {"supercategory": "kitchen", "id": 51, "name": "bowl"},
               {"supercategory": "food", "id": 52, "name": "banana"},
               {"supercategory": "food", "id": 53, "name": "apple"},
               {"supercategory": "food", "id": 54, "name": "sandwich"},
               {"supercategory": "food", "id": 55, "name": "orange"},
               {"supercategory": "food", "id": 56, "name": "broccoli"},
               {"supercategory": "food", "id": 57, "name": "carrot"},
               {"supercategory": "food", "id": 58, "name": "hot dog"},
               {"supercategory": "food", "id": 59, "name": "pizza"},
               {"supercategory": "food", "id": 60, "name": "donut"},
               {"supercategory": "food", "id": 61, "name": "cake"},
               {"supercategory": "furniture", "id": 62, "name": "chair"},
               {"supercategory": "furniture", "id": 63, "name": "couch"},
               {"supercategory": "furniture", "id": 64, "name": "potted plant"},
               {"supercategory": "furniture", "id": 65, "name": "bed"},
               {"supercategory": "furniture", "id": 67, "name": "dining table"},
               {"supercategory": "furniture", "id": 70, "name": "toilet"},
               {"supercategory": "electronic", "id": 72, "name": "tv"},
               {"supercategory": "electronic", "id": 73, "name": "laptop"},
               {"supercategory": "electronic", "id": 74, "name": "mouse"},
               {"supercategory": "electronic", "id": 75, "name": "remote"},
               {"supercategory": "electronic", "id": 76, "name": "keyboard"},
               {"supercategory": "electronic", "id": 77, "name": "cell phone"},
               {"supercategory": "appliance", "id": 78, "name": "microwave"},
               {"supercategory": "appliance", "id": 79, "name": "oven"},
               {"supercategory": "appliance", "id": 80, "name": "toaster"},
               {"supercategory": "appliance", "id": 81, "name": "sink"},
               {"supercategory": "appliance", "id": 82, "name": "refrigerator"},
               {"supercategory": "indoor", "id": 84, "name": "book"},
               {"supercategory": "indoor", "id": 85, "name": "clock"},
               {"supercategory": "indoor", "id": 86, "name": "vase"},
               {"supercategory": "indoor", "id": 87, "name": "scissors"},
               {"supercategory": "indoor", "id": 88, "name": "teddy bear"},
               {"supercategory": "indoor", "id": 89, "name": "hair drier"},
               {"supercategory": "indoor", "id": 90, "name": "toothbrush"}]

cls_dict = {}
for index, item in enumerate(classes):
    category_id = item['id']
    cls_dict[category_id] = index


def get_coco_cls_label(name):
    label_txt = open('/home/users/u5876230/coco/annotations/bbx/' + name + '.txt')
    label = label_txt.readlines()[0:]
    # print(name, label)
    # label_list = []

    multi_cls_lab = np.zeros((80), np.float32)

    for item in label:
        # category_index = int(item.split(',')[0].split(':')[1])
        category_index = int(item.split(' ')[2])

        class_index = cls_dict[category_index]
        # print(class_index)
        multi_cls_lab[class_index] = 1

    # multi_cls_lab = torch.from_numpy(multi_cls_lab)
    label_txt.close()
    return multi_cls_lab


def get_data_from_chunk_coco(chunk, args):
    # print(chunk)
    img_path = args.IMpath

    scale = np.random.uniform(0.7, 1.3)

    dim = args.crop_size
    images = np.zeros((dim, dim, 3, len(chunk)))
    saliency = np.zeros((dim, dim, len(chunk)))

    ori_images = np.zeros((dim, dim, 3, len(chunk)),dtype=np.uint8)
    croppings = np.zeros((dim, dim, len(chunk)))
    labels = []
    name_list = []

    for i, piece in enumerate(chunk):
        piece = piece.split('.')[0]
        cls_label = get_coco_cls_label(piece)
        assert(np.sum(cls_label)>0)

        # print(cls_label)
        labels.append(cls_label)

        name_list.append(piece)
        flip_p = np.random.uniform(0, 1)
        img_temp = cv2.imread(os.path.join(img_path, piece + '.jpg'))
        img_temp = cv2.cvtColor(img_temp,cv2.COLOR_BGR2RGB).astype(np.float)
        saliency_map_path = os.path.join('/home/users/u5876230/swin_sod/coco_saliency/', '{}.png'.format(piece))
        saliency_map = PIL.Image.open(saliency_map_path)
        saliency_map = np.asarray(saliency_map)
        # print(saliency_map.shape)
        # img_temp = scale_im(img_temp, scale)
        img_temp, saliency_map = RandomResizeLong2(img_temp, saliency_map, int(dim*0.9), int(dim/0.875))

        img_temp, saliency_map = flip2(img_temp,saliency_map, flip_p)

        img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
        img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
        img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225
        img_temp, cropping, saliency_map = RandomCrop2(img_temp,saliency_map, dim)

        ori_temp = np.zeros_like(img_temp)
        ori_temp[:, :, 0] = (img_temp[:, :, 0] * 0.229 + 0.485) * 255.
        ori_temp[:, :, 1] = (img_temp[:, :, 1] * 0.224 + 0.456) * 255.
        ori_temp[:, :, 2] = (img_temp[:, :, 2] * 0.225 + 0.406) * 255.
        ori_images[:, :, :, i] = ori_temp.astype(np.uint8)
        croppings[:,:,i] = cropping.astype(np.float32)

        images[:, :, :, i] = img_temp
        saliency[:, :,  i] = saliency_map

    images = images.transpose((3, 2, 0, 1))
    saliency = saliency.transpose((2, 0, 1))
    ori_images = ori_images.transpose((3, 2, 0, 1))
    images = torch.from_numpy(images).float()
    saliency = torch.from_numpy(saliency).float()
    labels = torch.from_numpy(np.array(labels))


    return images, ori_images, labels, croppings, name_list, saliency


def compute_cos(fts1, fts2):
    fts1_norm2 = torch.norm(fts1, 2, 1).view(-1, 1)
    fts2_norm2 = torch.norm(fts2, 2, 1).view(-1, 1)

    fts_cos = torch.div(torch.mm(fts1, fts2.t()), torch.mm(fts1_norm2, fts2_norm2.t()) + 1e-7)

    return fts_cos

def compute_dis_no_batch(seg, seg_feature):
    seg = torch.argmax(seg, dim=1, keepdim=True).view(seg.shape[0],1, -1)
    seg_no_batch = seg.permute(0,2,1).clone().view(-1,1)

    bg_label = torch.zeros_like(seg).float()

    bg_label[seg == 0] = 1
    bg_num = torch.sum(bg_label) + 1e-7

    seg_feature = seg_feature.view(seg_feature.shape[0], seg_feature.shape[1], -1)

    seg_feature_no_batch = seg_feature.permute(0, 2, 1).clone()
    seg_feature_no_batch = seg_feature_no_batch.view(-1, seg_feature.shape[1])

    seg_feature_bg = seg_feature * bg_label
    bg_num_batch = torch.sum(bg_label, dim=2)+1e-7
    seg_feature_bg_center = torch.sum(seg_feature_bg, dim=2) / bg_num_batch
    pixel_dis = 0

    bg_center_num = 0
    for batch_i in range(seg_feature.shape[0]):
        bg_num_batch_i = bg_num_batch[batch_i]
        bg_pixel_dis = 1-compute_cos(seg_feature[batch_i].transpose(1,0), seg_feature_bg_center[batch_i].unsqueeze(dim=0))
        if bg_num_batch_i>=1:
            pixel_dis += (torch.sum(bg_pixel_dis * bg_label[batch_i].transpose(1,0), dim=0)/ bg_num_batch_i)
        else:
            pixel_dis += 2*torch.ones([1]).cuda()

        bg_center_num+=1

    fg_center_num=0
    seg_feature_fg_center = torch.zeros([1, 1024])
    batch_num = 0
    for i in range(1, 21):
        class_label = torch.zeros_like(seg_no_batch).float()
        class_label[seg_no_batch == i] = 1
        class_num = torch.sum(class_label) + 1e-7
        batch_num += class_num
        if class_num < 1:
            continue
        else:
            seg_feature_class = seg_feature_no_batch * class_label
            seg_feature_class_center = torch.sum(seg_feature_class, dim=0, keepdim=True) / class_num
            fg_pixel_dis = 1-compute_cos(seg_feature_no_batch, seg_feature_class_center)
            pixel_dis += (torch.sum(fg_pixel_dis*class_label,dim=0)/ class_num)
            fg_center_num += 1
            if fg_center_num == 1:
                seg_feature_fg_center = seg_feature_class_center
            else:
                seg_feature_fg_center = torch.cat([seg_feature_fg_center, seg_feature_class_center], dim=0)

    pixel_dis = pixel_dis / (fg_center_num+bg_center_num)

    if batch_num >= 1 and torch.sum(bg_num) >= 1:

        fg_fg_cos = 1 + compute_cos(seg_feature_fg_center, seg_feature_fg_center)
        fg_bg_cos = 1 + compute_cos(seg_feature_fg_center, seg_feature_bg_center)

        fg_fg_cos = fg_fg_cos - torch.diag(torch.diag(fg_fg_cos))
        if fg_fg_cos.shape[0]>1:
            fg_fg_loss = torch.sum(fg_fg_cos) / (fg_fg_cos.shape[0] * (fg_fg_cos.shape[1] - 1))

        else:
            fg_fg_loss = torch.zeros([1]).cuda()
        fg_bg_loss = torch.sum(fg_bg_cos) / (fg_bg_cos.shape[0] * fg_bg_cos.shape[1])
        dis_loss = 0.5 * fg_fg_loss.cuda() + 0.5 * fg_bg_loss.cuda()

    elif torch.sum(bg_num) < 1:
        fg_norm2 = torch.norm(seg_feature_fg_center, 2, 1).view(-1, 1)

        fg_fg_cos = 1 + torch.div(torch.mm(seg_feature_fg_center, seg_feature_fg_center.t()),
                                  torch.mm(fg_norm2, fg_norm2.t()) + 1e-7)

        fg_fg_cos = fg_fg_cos - torch.diag(torch.diag(fg_fg_cos))

        if fg_fg_cos.shape[0]>1:
            fg_fg_loss = torch.sum(fg_fg_cos) / (fg_fg_cos.shape[0] * (fg_fg_cos.shape[1] - 1))

        else:
            fg_fg_loss = torch.zeros([1]).cuda()

        dis_loss = 0.5 * fg_fg_loss + 1

    else:
        dis_loss = torch.zeros([1]).cuda()

    return dis_loss.cuda()+pixel_dis.cuda()


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]

    # unknown classes
    r[label_mask == 255] = 255
    g[label_mask == 255] = 255
    b[label_mask == 255] = 255

    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def get_cityscapes_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


from PIL import Image
from tool.metrics import Evaluator

def _crf_with_alpha(pred_prob, ori_img):
    bgcam_score = pred_prob.cpu().data.numpy()
    crf_score = imutils.crf_inference_inf(ori_img, bgcam_score, labels=21)

    return crf_score


def validation(model, use_crf=False):
    model.eval()
    evaluator = Evaluator(num_class=21) 

    im_path = "/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages"
    img_list = open('voc12/val(id).txt').readlines()
    pred_softmax = torch.nn.Softmax(dim=0)
    with torch.no_grad():
        for index, i in enumerate(img_list):
            # print(index)
            # i = ((i.split('/'))[2])[0:-4]


            # print(os.path.join(im_path, i[:-1] + '.jpg'))
            img_temp = cv2.imread(os.path.join(im_path, i[:-1] + '.jpg'))
            target_path = os.path.join('/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/SegmentationClassAug', '{}.png'.format(i[:-1]))
            target = np.asarray(Image.open(target_path), dtype=np.int32)
            h, w, _ = img_temp.shape
            img_original = img_temp.astype(np.uint8)

            test_size = 256
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

            img_temp = cv2.resize(img_temp, (test_size, test_size))

            img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB).astype(np.float)
            img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
            img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
            img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225

            input = torch.from_numpy(img_temp[np.newaxis, :].transpose(0, 3, 1, 2)).float().cuda()

            _, output= model(input)

            # output = output[]
            # print(output.shape)

            # if h>=w:
            #     output = output[:,:,:,0:int(test_size*h/w)]
            # else:
            #     output = output[:,:,0:int(test_size*h/w), :]


            output = F.interpolate(output, (h, w),mode='bilinear',align_corners=False)
            output = torch.squeeze(output)
            pred_prob = pred_softmax(output)

            
            if use_crf:
                crf_la = _crf_with_alpha(pred_prob, img_original)
                crf_img = np.argmax(crf_la, 0)
                evaluator.add_batch(target, crf_img)
            else:
                output = torch.argmax(output,dim=0).cpu().numpy()
                evaluator.add_batch(target, output)

    mIoU = evaluator.Mean_Intersection_over_Union()
    return mIoU
        
