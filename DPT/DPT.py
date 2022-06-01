from numpy.core.numeric import roll
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


# def compute_rollout_attention(all_layer_matrices, start_layer=0):
#     # adding residual consideration
#     num_tokens = all_layer_matrices[0].shape[1]
#     batch_size = all_layer_matrices[0].shape[0]

#     eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
#     all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
#     # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
#     #                       for i in range(len(all_layer_matrices))]

#     joint_attention = all_layer_matrices[start_layer]
#     for i in range(start_layer+1, len(all_layer_matrices)):
#         joint_attention = all_layer_matrices[i].bmm(joint_attention)
#     return joint_attention


# def compute_rollout_attention_2(all_layer_matrices, start_layer=0):
#     # adding residual consideration
#     num_tokens = all_layer_matrices[0].shape[1]
#     batch_size = all_layer_matrices[0].shape[0]

#     eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
#     all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
#     # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
#     #                       for i in range(len(all_layer_matrices))]

#     joint_attention = all_layer_matrices[start_layer]
#     for i in range(start_layer+1, len(all_layer_matrices)):
#         joint_attention = all_layer_matrices[i]*(joint_attention)
#     return joint_attention

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

# class CBAM(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
#         super(CBAM, self).__init__()
#         modules_body = []
#         act = nn.ReLU(True)
#         for i in range(2):
#             modules_body.append(self.default_conv(gate_channels, gate_channels, 3, bias=True))
#             if i == 0: modules_body.append(act)
#         self.body = nn.Sequential(*modules_body)
#         self.gate_channels = gate_channels
#         self.mlp = nn.Sequential(
#             Flatten(),
#             nn.Linear(gate_channels, gate_channels // reduction_ratio),
#             nn.ReLU(),
#             nn.Linear(gate_channels // reduction_ratio, gate_channels)
#             )
#         self.pool_types = pool_types
    
#     def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
#         return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

#     def forward(self, x):
#         raw_x = x
#         x = self.body(x)
#         channel_att_sum = None
#         for pool_type in self.pool_types:
#             if pool_type=='avg':
#                 avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp( avg_pool )
#             elif pool_type=='max':
#                 max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp( max_pool )
#             elif pool_type=='lp':
#                 lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp( lp_pool )
#             elif pool_type=='lse':
#                 # LSE pool only
#                 lse_pool = logsumexp_2d(x)
#                 channel_att_raw = self.mlp( lse_pool )

#             if channel_att_sum is None:
#                 channel_att_sum = channel_att_raw
#             else:
#                 channel_att_sum = channel_att_sum + channel_att_raw

#         scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)

#         return x * scale + raw_x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        modules_body = []
        act = nn.ReLU(True)
        for i in range(2):
            modules_body.append(self.default_conv(channel, channel, 3, bias=True))
            if i == 0: modules_body.append(act)
        self.body = nn.Sequential(*modules_body)

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        raw_x = x
        x = self.body(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # print(x.shape, y.shape)
        at_x = x * y.expand_as(x)
        return at_x + raw_x

from DPT.blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)

class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x

class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone='vitb_rn50_384',
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
        use_pretrain=True,
        use_attention=False
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last
        self.attention = use_attention

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            'deitb16_384': [2,5,8,11],
            'deitb16_distil_384':[2,5,8,11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            use_pretrain,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        self.cam_module = SELayer(channel=256)
        self.scratch.output_conv = head

        # self.embedding_layer = nn.Sequential(
        #         nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(256),
        #         nn.ReLU())


        # classification head
        self.cls_head = nn.Linear(768, self.num_class)

    def forward(self, x):
        raw_x = x
        x_size = x.size()
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4, x_cls, _ = forward_vit(self.pretrained, x)

        x_cls = layer_4.clone()
        x_cls = F.avg_pool2d(x_cls, kernel_size=(x_cls.size(2), x_cls.size(3)), padding=0)
        x_cls = self.cls_head(x_cls.squeeze(3).squeeze(2))

        # x_cls = self.cls_head(x_cls)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        path_1 = self.cam_module(path_1)
        out = self.scratch.output_conv(path_1)

        return x_cls, out
    
    def forward_seg(self,x):
        raw_x = x
        x_size = x.size()
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4, x_cls, _ = forward_vit(self.pretrained, x)

        # x_cls = self.cls_head(x_cls)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        path_1 = self.cam_module(path_1)

        out = self.scratch.output_conv(path_1)

        return out


    def forward_cls(self, x):
        x_size = x.size()
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4, _, _ = forward_vit(self.pretrained, x)

        x_cls = layer_4.clone()
        x_cls = F.avg_pool2d(x_cls, kernel_size=(x_cls.size(2), x_cls.size(3)), padding=0)
        x_cls = self.cls_head(x_cls.squeeze(3).squeeze(2))

        return x_cls

    # def forward_cam(self, x):
    #     if self.channels_last == True:
    #         x.contiguous(memory_format=torch.channels_last)

    #     layer_1, layer_2, layer_3, layer_4, _, _ = forward_vit(self.pretrained, x)
        
    #     x_cls = layer_4.clone()
    #     x_cls = F.avg_pool2d(x_cls, kernel_size=(x_cls.size(2), x_cls.size(3)), padding=0)
    #     x_cls = self.cls_head(x_cls.squeeze(3).squeeze(2))
    #     # x_cls = self.cls_head(x_cls)

    #     x_cam = layer_4.clone()
    #     cam = self.cls_head(x_cam.permute(0,2,3,1))
    #     cam = cam.permute(0,3,1,2)
    #     cam = F.relu(cam)
        
    #     return x_cls, cam

class DPTSegmentationModel(DPT):
    def __init__(self, num_classes, backbone_name, path=None, **kwargs):
        self.num_class = num_classes

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes+1, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        backbone_dict = {"vitb_hybrid": "vitb_rn50_384",
            'vitb':"vitb16_384",
            'deit':'deitb16_384',
            'deit_distilled':'deitb16_distil_384',
            'vitl':"vitl16_384",
        }

        cur_backbone = backbone_dict[backbone_name]
        self.cur_backbone = cur_backbone
        print('cur_backbone:', cur_backbone)
        super().__init__(head, backbone=cur_backbone, **kwargs)

        if path is not None:
            self.load(path)
    
    # "GETAM" cam * gradient^2
    def generate_cam_2(self, batch, start_layer=0):
        cam_list = []
        attn_list = []
        grad_list = []
        for blk in self.pretrained.model.blocks:
            grad = blk.attn.get_attn_gradients()
            cam = blk.attn.get_attn()
            attn_list.append(torch.mean(cam, dim = 1))
            grad_list.append(torch.mean(grad, dim = 1))
            cam = cam[batch].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[batch].reshape(-1, grad.shape[-1], grad.shape[-1])
            
            cam = grad * cam 
            cam = cam.clamp(min=0).mean(dim=0)
            
            positive_grad = grad.clamp(min=0).mean(dim=0)
            cam = cam * positive_grad

            cam_list.append(cam.unsqueeze(0))

        cam_list = cam_list[start_layer:]
        cams = torch.stack(cam_list).sum(dim=0)
        if self.cur_backbone == 'deitb16_distil_384':
            cls_cam = torch.relu(cams[:, 0, 2:])
        else:
            cls_cam = torch.relu(cams[:, 0, 1:])

        return cls_cam, attn_list, cam_list
        






