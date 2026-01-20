from emrmodel.mask_rcnn import MaskRCNN
from emrmodel.stacked_fpn_backbone import Stacked_Resnet50FPN_Backbone
from emrmodel.early_mlp_fusion import SliceSEFusion, SliceSEFusionFixedWindow, IdentityFusion
from torchvision.models.detection.backbone_utils import BackboneWithFPN, _validate_trainable_layers, _resnet_fpn_extractor
from torchvision.models.resnet import resnet50, ResNet50_Weights
import torch
from torchvision.ops import misc as misc_nn_ops
from typing import Optional
import torch.nn as nn



class ExtendedMaskRCNN(MaskRCNN):
    def __init__(self, num_slices_per_batch = 3, backbone=None, num_classes=2, min_size=800, max_size=1333, image_mean=None, image_std=None, rpn_anchor_generator=None, rpn_head=None, rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000, rpn_nms_thresh=0.7, rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3, rpn_batch_size_per_image=256, rpn_positive_fraction=0.5, rpn_score_thresh=0, box_roi_pool=None, box_head=None, box_predictor=None, box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100, box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5, box_batch_size_per_image=512, box_positive_fraction=0.25, bbox_reg_weights=None, mask_roi_pool=None, mask_head=None, mask_predictor=None, early_mlp_fusion="None", early_mlp_reduction=16, early_mlp_bias=None, **kwargs):

        self.training = True
        self.num_classes = num_classes

        if early_mlp_fusion == "None":
            if backbone == None: # see maskrcnn_resnet50_fpn in torchvision/models/detection/mask_rcnn.py
                norm_layer = misc_nn_ops.FrozenBatchNorm2d
                trainable_backbone_layers = _validate_trainable_layers(True, 5, max_value=5, default_value=3) # trainable backbone layers is passed as 5
                backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, progress=True, norm_layer=norm_layer)
                backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
                in_channels = num_slices_per_batch # number of input slices
                backbone.body.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)   
        else:
            if backbone == None:
                backbone = Stacked_Resnet50FPN_Backbone(num_slices=num_slices_per_batch)     
       
        if image_mean == None:
            image_mean = [0 for _ in range(num_slices_per_batch)]
        if image_std == None:
            image_std = [1 for _ in range(num_slices_per_batch)]
        super().__init__(
            backbone = backbone,
            num_classes=self.num_classes,
            # transform parameters
            min_size=min_size,
            max_size=max_size,
            image_mean=image_mean,
            image_std=image_std,
            # RPN parameters
            rpn_anchor_generator=rpn_anchor_generator,
            rpn_head=rpn_head,
            rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
            rpn_nms_thresh=rpn_nms_thresh,
            rpn_fg_iou_thresh=rpn_fg_iou_thresh,
            rpn_bg_iou_thresh=rpn_bg_iou_thresh,
            rpn_batch_size_per_image=rpn_batch_size_per_image,
            rpn_positive_fraction=rpn_positive_fraction,
            rpn_score_thresh=rpn_score_thresh,
            # Box parameters
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=box_detections_per_img,
            box_fg_iou_thresh=box_fg_iou_thresh,
            box_bg_iou_thresh=box_bg_iou_thresh,
            box_batch_size_per_image=box_batch_size_per_image,
            box_positive_fraction=box_positive_fraction,
            bbox_reg_weights=bbox_reg_weights,
            # Mask parameters
            mask_roi_pool=mask_roi_pool,
            mask_head=mask_head,
            mask_predictor=mask_predictor,
            **kwargs,
        )

        early_mlp_fusion_params = {
            'num_slices': num_slices_per_batch,
            'channels': backbone.out_channels,
            'reduction': early_mlp_reduction,
            'init_bias': early_mlp_bias

        }

        if early_mlp_fusion == "None":
            self.early_mlp_fusion_module = IdentityFusion(**early_mlp_fusion_params)
        elif early_mlp_fusion == "Global":
            self.early_mlp_fusion_module = SliceSEFusion(**early_mlp_fusion_params)
        elif early_mlp_fusion == "Windowed":
            self.early_mlp_fusion_module = SliceSEFusionFixedWindow(**early_mlp_fusion_params)


        self._init_weights_non_backbone()
        return
    

    def _init_weights_non_backbone(self):
        for name, m in self.named_modules():
            if name.startswith("backbone"):
                continue

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


