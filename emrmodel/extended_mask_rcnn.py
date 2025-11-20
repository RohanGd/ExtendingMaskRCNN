from emrmodel.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN, _validate_trainable_layers, _resnet_fpn_extractor
from torchvision.models.resnet import resnet50, ResNet50_Weights
import torch
from torchvision.ops import misc as misc_nn_ops
from typing import Optional



class ExtendedMaskRCNN(MaskRCNN):
    def __init__(self, num_slices_per_batch = 3, backbone=None, num_classes=2, min_size=800, max_size=1333, image_mean=None, image_std=None, rpn_anchor_generator=None, rpn_head=None, rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000, rpn_nms_thresh=0.7, rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3, rpn_batch_size_per_image=256, rpn_positive_fraction=0.5, rpn_score_thresh=0, box_roi_pool=None, box_head=None, box_predictor=None, box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100, box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5, box_batch_size_per_image=512, box_positive_fraction=0.25, bbox_reg_weights=None, mask_roi_pool=None, mask_head=None, mask_predictor=None, **kwargs):

        self.training = True
        self.num_classes = num_classes

        if backbone == None: # see maskrcnn_resnet50_fpn in torchvision/models/detection/mask_rcnn.py
            norm_layer = misc_nn_ops.FrozenBatchNorm2d
            trainable_backbone_layers = _validate_trainable_layers(True, None, max_value=5, default_value=3) # trainable backbone layers is passed as None
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, progress=True, norm_layer=norm_layer)
            backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
            in_channels = num_slices_per_batch # number of input slices
            backbone.body.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)        
        
        super().__init__(
            backbone = backbone,
            num_classes=self.num_classes,
            # transform parameters
            min_size=min_size,
            max_size=max_size,
            image_mean=None,
            image_std=None,
            # RPN parameters
            rpn_anchor_generator=None,
            rpn_head=None,
            rpn_pre_nms_top_n_train=2000,
            rpn_pre_nms_top_n_test=1000,
            rpn_post_nms_top_n_train=2000,
            rpn_post_nms_top_n_test=1000,
            rpn_nms_thresh=0.7,
            rpn_fg_iou_thresh=0.7,
            rpn_bg_iou_thresh=0.3,
            rpn_batch_size_per_image=256,
            rpn_positive_fraction=0.5,
            rpn_score_thresh=0.0,
            # Box parameters
            box_roi_pool=None,
            box_head=None,
            box_predictor=None,
            box_score_thresh=0.05,
            box_nms_thresh=0.5,
            box_detections_per_img=100,
            box_fg_iou_thresh=0.5,
            box_bg_iou_thresh=0.5,
            box_batch_size_per_image=512,
            box_positive_fraction=0.25,
            bbox_reg_weights=None,
            # Mask parameters
            mask_roi_pool=None,
            mask_head=None,
            mask_predictor=None,
            **kwargs,
        )

        return