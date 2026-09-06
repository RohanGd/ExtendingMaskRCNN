import os
import warnings
import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator
from emrmodel.extended_mask_rcnn import ExtendedMaskRCNN


def build_anchor_generator(sizes_str, ratios_str):
    """Build an AnchorGenerator from the [MODEL] anchor_sizes / anchor_aspect_ratios keys.

    anchor_sizes is one FPN level per "|", sizes within a level comma-separated, e.g.
        anchor_sizes = 12,16,24 | 24,32,48 | 32,48,64 | 48,64,96 | 64,96,128
    anchor_aspect_ratios is a single comma-separated list applied to every level.

    Returns None when anchor_sizes is unset, so the model falls back to
    _default_anchorgen() in emrmodel/faster_rcnn.py.

    Each level must carry the SAME number of sizes: RPNHead emits one conv with a fixed
    channel count for all levels, so a ragged config (e.g. 4 sizes on P2 but 3 on P4)
    fails at box decode with "shape [N,-1] is invalid for input of size M".
    """
    if not sizes_str or not str(sizes_str).strip():
        return None

    sizes = tuple(
        tuple(int(v) for v in level.split(",") if v.strip())
        for level in str(sizes_str).split("|")
    )
    per_level = {len(level) for level in sizes}
    if len(per_level) != 1:
        raise ValueError(
            f"anchor_sizes must have the same number of sizes on every FPN level, got "
            f"{[len(level) for level in sizes]}. RPNHead cannot handle a ragged config."
        )

    ratios = tuple(float(v) for v in str(ratios_str).split(",") if v.strip()) \
        if ratios_str and str(ratios_str).strip() else (0.75, 1.0, 1.5)
    return AnchorGenerator(sizes, (ratios,) * len(sizes))

class ModelBuilder:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.start_epoch = 0
        self.optimizer_state_dict = None

    def load_model(self, dataset_name="None"):
        num_slices_per_batch = self.cfg.get_int("MODEL", "num_slices_per_batch", 3)
        min_size = self.cfg.get_int("MODEL", "min_size", 512)
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
        rpn_post_nms_top_n_train = self.cfg.get_int("MODEL", "rpn_post_nms_top_n_train", 2000)
        rpn_post_nms_top_n_test = self.cfg.get_int("MODEL", "rpn_post_nms_top_n_test", 1000)
        # Sampling budgets. Both cap how many objects can contribute a gradient per image:
        # rpn_batch_size_per_image * rpn_positive_fraction positive anchors, and
        # box_batch_size_per_image * box_positive_fraction positive RoIs. On crowded
        # datasets (ATAS runs 110-200 instances/slice) the torchvision defaults of
        # 256*0.5 = 128 and 512*0.25 = 128 sit below the object count, so most cells are
        # never sampled -- raise these alongside box_detections_per_img.
        rpn_batch_size_per_image = self.cfg.get_int("MODEL", "rpn_batch_size_per_image", 256)
        box_batch_size_per_image = self.cfg.get_int("MODEL", "box_batch_size_per_image", 512)
        box_positive_fraction = self.cfg.get_float("MODEL", "box_positive_fraction", 0.25)
        rpn_fg_iou_thresh = self.cfg.get_float("MODEL", "rpn_fg_iou_thresh", 0.7)
        rpn_bg_iou_thresh = self.cfg.get_float("MODEL", "rpn_bg_iou_thresh", 0.3)
        box_fg_iou_thresh = self.cfg.get_float("MODEL", "box_fg_iou_thresh", 0.5)
        box_bg_iou_thresh = self.cfg.get_float("MODEL", "box_bg_iou_thresh", 0.5)
        early_mlp_fusion = self.cfg.get("MODEL", "early_mlp_fusion", "None")
        early_mlp_reduction = self.cfg.get_int("MODEL", "early_mlp_reduction", 16)
        early_mlp_bias = self.cfg.get("MODEL", "early_mlp_bias", "None")
        roi_heads_fusion = self.cfg.get("MODEL", "roi_heads_fusion", "None")
        backbone = self.cfg.get("MODEL", "backbone", None) # None (default resnet50) or "Swin"
        # Anchor sizes are bound to FPN levels, whose strides are fixed at 4/8/16/32/64, so
        # what governs RPN recall is each anchor's size-to-stride ratio -- not its absolute
        # size. Lowering the whole ladder pushes every size onto a relatively coarser level
        # and makes matching worse; carrying several sizes per level is what helps. On ATAS
        # (median cell 20px at 512) the 1-size-per-level default matches only 34% of GT
        # boxes at >=0.7 IoU, versus 62% for 3 sizes per level.
        anchor_generator = build_anchor_generator(
            self.cfg.get("MODEL", "anchor_sizes", None),
            self.cfg.get("MODEL", "anchor_aspect_ratios", None),
        )


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
            'rpn_post_nms_top_n_train': rpn_post_nms_top_n_train,
            'rpn_post_nms_top_n_test': rpn_post_nms_top_n_test,
            'rpn_batch_size_per_image': rpn_batch_size_per_image,
            'box_batch_size_per_image': box_batch_size_per_image,
            'box_positive_fraction': box_positive_fraction,
            'rpn_fg_iou_thresh': rpn_fg_iou_thresh,
            'rpn_bg_iou_thresh': rpn_bg_iou_thresh,
            'box_fg_iou_thresh': box_fg_iou_thresh,
            'box_bg_iou_thresh': box_bg_iou_thresh,
            'rpn_positive_fraction': rpn_positive_fraction,
            'early_mlp_fusion': early_mlp_fusion,
            'early_mlp_reduction': early_mlp_reduction,
            'early_mlp_bias': early_mlp_bias,
            'roi_heads_fusion': roi_heads_fusion,
            'backbone': backbone,
            'rpn_anchor_generator': anchor_generator,
        }

        self.logger.info(f"MODEL PARAMS: {model_params}")
        
        model = ExtendedMaskRCNN(**model_params)

        self.ckpt_path = self.cfg.get("LOOP", "ckpt_path", "")
        self.backbone_ckpt_path = self.cfg.get("LOOP", "backbone_ckpt_path", "")
        self.freeze_backbone = self.cfg.get_bool("LOOP", "freeze_backbone", False)
        # When set, backbone_ckpt_path also warm-starts (and, if freeze_backbone is
        # set, freezes) the RPN head and box head/predictor -- these operate on the
        # center slice's features only (see generalized_rcnn.py/roi_heads.py), so
        # they're architecturally identical to a single-slice (base_n1) checkpoint's
        # RPN/box head and can be reused as-is. Only the mask head, mask predictor,
        # and mask_features_fusion module (the parts that actually differ for a
        # fusion model) are left randomly initialized and trainable.
        self.warm_start_rpn_box = self.cfg.get_bool("LOOP", "warm_start_rpn_box", False)

        if self.ckpt_path and self.backbone_ckpt_path:
            raise ValueError(
                "[LOOP] has both ckpt_path and backbone_ckpt_path set -- set only one."
            )

        def _model_state_dict(path):
            checkpoint = torch.load(path, weights_only=True)
            # older checkpoints are a bare model.state_dict(); newer ones are
            # {"model_state_dict", "optimizer_state_dict", "epoch"}
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                return checkpoint["model_state_dict"], checkpoint
            return checkpoint, None

        if self.backbone_ckpt_path:
            if os.path.exists(self.backbone_ckpt_path):
                model_state_dict, _ = _model_state_dict(self.backbone_ckpt_path)
                warm_start_prefixes = ("backbone.",)
                if self.warm_start_rpn_box:
                    warm_start_prefixes += ("rpn.", "roi_heads.box_head.", "roi_heads.box_predictor.")
                warm_start_state_dict = {k: v for k, v in model_state_dict.items() if k.startswith(warm_start_prefixes)}
                model.load_state_dict(warm_start_state_dict, strict=False)
                self.logger.info(f"Loaded weights ({warm_start_prefixes}) from: {self.backbone_ckpt_path}")
            else:
                self.logger.warning(f"backbone_ckpt_path not found: {self.backbone_ckpt_path}")
        elif self.ckpt_path:
            if os.path.exists(self.ckpt_path):
                model_state_dict, full_checkpoint = _model_state_dict(self.ckpt_path)
                model.load_state_dict(model_state_dict)
                if full_checkpoint is not None:
                    self.start_epoch = full_checkpoint.get("epoch", 0)
                    self.optimizer_state_dict = full_checkpoint.get("optimizer_state_dict")
                self.logger.info(f"Loaded full model: {self.ckpt_path}, resuming from epoch {self.start_epoch}")
            else:
                self.logger.warning(f"Checkpoint not found: {self.ckpt_path}")
        else:
            self.logger.info(f"Created model with config: {self.cfg.path}")

        if self.freeze_backbone:
            freeze_prefixes = ("backbone.",)
            if self.warm_start_rpn_box:
                freeze_prefixes += ("rpn.", "roi_heads.box_head.", "roi_heads.box_predictor.")
            for name, param in model.named_parameters():
                if name.startswith(freeze_prefixes):
                    param.requires_grad = False
            self.logger.info(f"Frozen: {freeze_prefixes}")

        self.logger.info(f"PRINTING MODEL ARCHITECTURE: {model}")
        return model

    def build_optimizer(self, model):
        lr = self.cfg.get_float("LOOP", "learning_rate", 1e-4)
        wd = self.cfg.get_float("LOOP", "weight_decay", 1e-4)

        backbone_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": lr * 1.0},
            {"params": other_params, "lr": lr},
        ]

        # for name, param in model.named_parameters():
        #     if "norm" in name or "bn" in name:
        #         param_groups.append({"params": [param], "weight_decay": 0.0, "lr": lr})


        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)

        if self.optimizer_state_dict is not None:
            optimizer.load_state_dict(self.optimizer_state_dict)
            self.logger.info(f"Resumed optimizer state from: {self.ckpt_path}")

        return optimizer
