import os
import warnings
import torch
from emrmodel.extended_mask_rcnn import ExtendedMaskRCNN

class ModelBuilder:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.start_epochs = 0

    def load_model(self, dataset_name="None"):
        num_slices_per_batch = self.cfg.get_int("MODEL", "num_slices_per_batch", 3)
        min_size = self.cfg.get_int("MODEL", "min_size", 800)
        max_size = self.cfg.get_int("MODEL", "max_size", 1333)
        num_classes = self.cfg.get_int("MODEL", "num_classes", 2)
        rpn_positive_fraction = self.cfg.get_float("MODEL", "rpn_positive_fraction", 0.5)
        image_mean = self.cfg.get("MODEL", "image_mean", None)
        image_std = self.cfg.get("MODEL", "image_std", None)
        rpn_nms_thresh = self.cfg.get_float("MODEL", "rpn_nms_thresh", 0.7)
        box_score_thresh = self.cfg.get_float("MODEL", "box_score_thresh", 0.05)
        rpn_pre_nms_top_n_train = self.cfg.get_int("MODEL", "rpn_pre_nms_top_n_train", 2000)
        rpn_pre_nms_top_n_test = self.cfg.get_int("MODEL", "rpn_pre_nms_top_n_test", 1000)
        box_nms_thresh = self.cfg.get_float("MODEL", "box_nms_thresh", 0.5)
        box_detections_per_img = self.cfg.get_int("MODEL", "box_detections_per_img", 100)
        rpn_fg_iou_thresh = self.cfg.get_float("MODEL", "rpn_fg_iou_thresh", 0.7)
        rpn_bg_iou_thresh = self.cfg.get_float("MODEL", "rpn_bg_iou_thresh", 0.3)
        box_fg_iou_thresh = self.cfg.get_float("MODEL", "box_fg_iou_thresh", 0.5)
        box_bg_iou_thresh = self.cfg.get_float("MODEL", "box_bg_iou_thresh", 0.5)

        model_params = {
            'num_slices_per_batch': num_slices_per_batch,
            'num_classes': num_classes,
            'min_size': min_size,
            'max_size': max_size,
            'image_mean': image_mean,
            'image_std': image_std,
            'rpn_nms_thresh': rpn_nms_thresh,
            'box_score_thresh': box_score_thresh,
            'rpn_pre_nms_top_n_train': rpn_pre_nms_top_n_train,
            'rpn_pre_nms_top_n_test': rpn_pre_nms_top_n_test,
            'box_nms_thresh': box_nms_thresh,
            'box_detections_per_img': box_detections_per_img,
            'rpn_fg_iou_thresh': rpn_fg_iou_thresh,
            'rpn_bg_iou_thresh': rpn_bg_iou_thresh,
            'box_fg_iou_thresh': box_fg_iou_thresh,
            'box_bg_iou_thresh': box_bg_iou_thresh,
            'rpn_positive_fraction': rpn_positive_fraction
        }
        
        model = ExtendedMaskRCNN(**model_params)

        # Load checkpoint only if start_epochs != 0
        self.ckpt_path = self.cfg.get("LOOP", "ckpt_path", "no checkpoint path specified")

        if self.ckpt_path != "":
            if os.path.exists(self.ckpt_path):
                checkpoint = torch.load(self.ckpt_path, weights_only=True)
                model.load_state_dict(checkpoint)
                self.logger.info(f"Loaded model: {self.ckpt_path}")
            else:
                self.logger.warning(f"Checkpoint not found: {self.ckpt_path}")
        else:
            self.logger.info(f"Created model.")
        return model

    def build_optimizer(self, model):
        lr = self.cfg.get_float("LOOP", "learning_rate", 1e-4)
        wd = self.cfg.get_float("LOOP", "weight_decay", 1e-4)
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
