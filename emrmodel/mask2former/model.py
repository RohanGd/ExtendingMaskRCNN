"""
EMRMask2Former: Mask2Former for 2.5D instance segmentation, with the same three slice-fusion regimes as
ExtendedMaskRCNN and the same input/output contract, so training_loop.py / testing_loop.py / save_preds /
emrMetrics2D work unchanged.

    channel  (early_mlp_fusion=None, roi_heads_fusion=None): one backbone with in_channels=S; the S slices are
             fused at the input conv.
    early    (early_mlp_fusion=Global|GlobalPerFPN|Pixel|PixelPerFPN): one 1-channel backbone shared across
             slices; per-slice maps are fused inside the pixel decoder right after each level's projection to
             256 channels (see pixel_decoder.py), by the same modules ExtendedMaskRCNN uses before its RPN.
    late     (roi_heads_fusion=mean|only_center|conv3d|SE): shared 1-channel backbone *and* pixel decoder per
             slice; the transformer decoder attends to the center slice's multi-scale features (as the RPN/box
             head use the center slice in ExtendedMaskRCNN), and the S per-slice mask_features are collapsed by
             emrmodel/mask_features_fusion.py before the mask einsum. Fusion here is per image, not per instance.

Per-slice backbones run as one batch of B*S single-channel images rather than a Python loop over slices.
"""
import math
from collections import OrderedDict

import torch
from torch import Tensor, nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import masks_to_boxes
from torchvision.ops import misc as misc_nn_ops

from emrmodel.early_mlp_fusion import (SliceSEFusion, SliceSEFusionPerFPN, SlicePixelAttention,
                                       SlicePixelAttentionPerFPN, SliceSEFusionFixedWindow)
from emrmodel.mask_features_fusion import get_mask_features_fusion
from emrmodel.swin_fpn_backbone import SwinStageExtractor, SWIN_T_STAGE_CHANNELS
from emrmodel.mask2former.config import M2FConfig
from emrmodel.mask2former.criterion import Mask2FormerLoss
from emrmodel.mask2former.pixel_decoder import (Mask2FormerPixelDecoder,
                                                Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention)
from emrmodel.mask2former.transformer_decoder import (Mask2FormerTransformerModule,
                                                      Mask2FormerMaskedAttentionDecoderLayer)

EARLY_FUSION_MODULES = {
    "Global": SliceSEFusion,
    "GlobalPerFPN": SliceSEFusionPerFPN,
    "Windowed": SliceSEFusionFixedWindow,
    "Pixel": SlicePixelAttention,
    "PixelPerFPN": SlicePixelAttentionPerFPN,
}

RESNET50_STAGE_CHANNELS = [256, 512, 1024, 2048]


class M2FBackbone(nn.Module):
    """ImageNet-initialized ResNet50 or Swin-T returning the 4 stage maps (strides 4, 8, 16, 32), high -> low
    resolution, with the stem conv swapped for `in_channels` inputs -- the same bodies (and the same random
    re-init of the stem) the Mask R-CNN backbones in this repo use, minus the FPN, which Mask2Former's pixel
    decoder replaces."""

    def __init__(self, in_channels: int, kind: str = None, stem_init: str = "random"):
        super().__init__()
        if stem_init not in ("random", "pretrained"):
            raise ValueError(f"stem_init must be random or pretrained, got {stem_init!r}")
        if kind == "Swin":
            if stem_init != "random":
                raise ValueError("stem_init = pretrained is only implemented for the ResNet50 backbone")
            self.body = SwinStageExtractor(in_channels)
            self.channels = list(SWIN_T_STAGE_CHANNELS)
        elif kind is None or kind == "None":
            # same as ExtendedMaskRCNN's channel-fusion resnet: FrozenBatchNorm, all layers trainable
            net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
            rgb_weight = net.conv1.weight.detach()
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if stem_init == "pretrained":
                # a grayscale slice fed as RGB hits the sum of the 3 colour filters; split that over the S input
                # channels so S identical slices respond like one. Keeps the ImageNet stem (and the FrozenBN
                # stats after it) meaningful, which matters at backbone_lr_mult 0.1 where a random stem barely trains
                with torch.no_grad():
                    net.conv1.weight.copy_(rgb_weight.sum(1, keepdim=True).repeat(1, in_channels, 1, 1) / in_channels)
            self.body = IntermediateLayerGetter(net, return_layers={"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"})
            self.channels = list(RESNET50_STAGE_CHANNELS)
        else:
            raise ValueError(f"Unknown backbone for Mask2Former: {kind!r} (expected unset or 'Swin')")

    def forward(self, x: Tensor):
        return list(self.body(x).values())


class EMRMask2Former(nn.Module):
    def __init__(self, num_slices_per_batch=3, backbone=None, num_classes=2, image_mean=None, image_std=None,
                 early_mlp_fusion="None", early_mlp_reduction=16, early_mlp_bias=None, roi_heads_fusion="None",
                 box_score_thresh=0.05, box_detections_per_img=100, num_queries=100, decoder_layers=10,
                 class_weight=2.0, mask_weight=5.0, dice_weight=5.0, no_object_weight=0.1, train_num_points=12544,
                 overlap_thresh=0.0, stem_init="random", **kwargs):
        super().__init__()
        self.num_slices = num_slices_per_batch
        self.num_classes = num_classes
        self.early_mlp_fusion = early_mlp_fusion
        self.roi_heads_fusion = roi_heads_fusion
        self.score_thresh = box_score_thresh
        self.detections_per_img = box_detections_per_img
        self.overlap_thresh = overlap_thresh  # 0 = off; Mask2Former's panoptic inference uses 0.8

        if early_mlp_fusion != "None" and roi_heads_fusion != "None":
            raise ValueError("Set only one of early_mlp_fusion / roi_heads_fusion.")
        if early_mlp_fusion != "None" and early_mlp_fusion not in EARLY_FUSION_MODULES:
            raise ValueError(f"Unknown early_mlp_fusion: {early_mlp_fusion!r}")
        self.per_slice = early_mlp_fusion != "None" or roi_heads_fusion != "None"

        self.config = M2FConfig(
            num_labels=num_classes - 1,
            num_queries=num_queries,
            decoder_layers=decoder_layers,
            class_weight=class_weight,
            mask_weight=mask_weight,
            dice_weight=dice_weight,
            no_object_weight=no_object_weight,
            train_num_points=train_num_points,
        )

        # same normalization convention as ExtendedMaskRCNN: per-slice mean/std, default identity
        in_channels = 1 if self.per_slice else num_slices_per_batch
        image_mean = image_mean if image_mean is not None else [0.0] * num_slices_per_batch
        image_std = image_std if image_std is not None else [1.0] * num_slices_per_batch
        self.register_buffer("image_mean", torch.as_tensor(image_mean, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.as_tensor(image_std, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)

        self.backbone = M2FBackbone(in_channels, backbone, stem_init)
        self.pixel_decoder = Mask2FormerPixelDecoder(self.config, feature_channels=self.backbone.channels)
        self.transformer_module = Mask2FormerTransformerModule(in_features=self.config.feature_size, config=self.config)
        self.class_predictor = nn.Linear(self.config.hidden_dim, self.config.num_labels + 1)
        self.criterion = Mask2FormerLoss(self.config)

        if early_mlp_fusion != "None":
            self.early_mlp_fusion_module = EARLY_FUSION_MODULES[early_mlp_fusion](
                num_slices=num_slices_per_batch, channels=self.config.feature_size,
                reduction=early_mlp_reduction, init_bias=early_mlp_bias,
            )
        else:
            self.early_mlp_fusion_module = None
        self.mask_features_fusion = get_mask_features_fusion(roi_heads_fusion, num_slices_per_batch) if roi_heads_fusion != "None" else None

        self._init_weights_non_backbone()
        self.last_loss_terms = {}

    # ------------------------------------------------------------------ init
    def _init_weights_non_backbone(self):
        """Mirrors Mask2FormerPreTrainedModel._init_weights for the forked heads; the backbone keeps ImageNet weights
        and the fusion modules keep their own init. Applied post-order like PreTrainedModel.post_init, so the
        per-layer overrides win over the generic Linear/Embedding init."""
        std = self.config.init_std
        xavier_std = self.config.init_xavier_std

        def init_fn(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            if isinstance(module, Mask2FormerTransformerModule):
                for input_projection in module.input_projections:
                    if not isinstance(input_projection, nn.Sequential):
                        nn.init.xavier_uniform_(input_projection.weight, gain=xavier_std)
                        nn.init.constant_(input_projection.bias, 0)
            elif isinstance(module, Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention):
                nn.init.constant_(module.sampling_offsets.weight, 0.0)
                thetas = torch.arange(module.n_heads, dtype=torch.int64).float() * (2.0 * math.pi / module.n_heads)
                grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
                grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(module.n_heads, 1, 1, 2)
                grid_init = grid_init.repeat(1, module.n_levels, module.n_points, 1)
                for i in range(module.n_points):
                    grid_init[:, :, i, :] *= i + 1
                with torch.no_grad():
                    module.sampling_offsets.bias.copy_(grid_init.view(-1))
                nn.init.constant_(module.attention_weights.weight, 0.0)
                nn.init.constant_(module.attention_weights.bias, 0.0)
                nn.init.xavier_uniform_(module.value_proj.weight)
                nn.init.constant_(module.value_proj.bias, 0.0)
                nn.init.xavier_uniform_(module.output_proj.weight)
                nn.init.constant_(module.output_proj.bias, 0.0)
            elif isinstance(module, Mask2FormerMaskedAttentionDecoderLayer):
                for p in module.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p, gain=xavier_std)
                nn.init.zeros_(module.cross_attn.in_proj_bias)
            elif isinstance(module, Mask2FormerPixelDecoder):
                nn.init.zeros_(module.level_embed)

        for head in (self.pixel_decoder, self.transformer_module, self.class_predictor):
            head.apply(init_fn)

    # ------------------------------------------------------------------ forward
    def _pixel_level(self, images: Tensor):
        """images (B, S, H, W), normalized -> (mask_features (B, C, H/4, W/4), 3 multi-scale maps)."""
        B, S, H, W = images.shape
        if not self.per_slice:
            return self.pixel_decoder(self.backbone(images))

        per_slice = self.backbone(images.reshape(B * S, 1, H, W))  # 4 levels of (B*S, C, h, w)

        if self.early_mlp_fusion_module is not None:
            # level -> list of S (B, C, h, w), fused inside the pixel decoder after projection
            features = [list(f.view(B, S, *f.shape[1:]).unbind(dim=1)) for f in per_slice]
            return self.pixel_decoder(features, slice_fusion=self.early_mlp_fusion_module)

        # late fusion: per-slice pixel decoder, center-slice memory for the queries, fused per-pixel embedding
        mask_features, multi_scale = self.pixel_decoder(per_slice)
        center = S // 2
        multi_scale = tuple(m.view(B, S, *m.shape[1:])[:, center] for m in multi_scale)
        mask_features = mask_features.view(B, S, *mask_features.shape[1:]).permute(0, 2, 1, 3, 4)  # (B, C, S, h, w)
        return self.mask_features_fusion(mask_features), multi_scale

    def forward(self, images: Tensor, targets=None):
        """
        images: (B, S, H, W). In train mode returns a loss dict whose values sum to the Mask2Former loss; in eval
        mode returns per-image dicts {masks (N,1,H,W) probs, scores (N,), labels (N,), boxes (N,4)}.
        """
        if self.training and targets is None:
            raise ValueError("targets should be passed in training mode")
        H, W = images.shape[-2:]
        images = (images - self.image_mean) / self.image_std

        mask_features, multi_scale = self._pixel_level(images)
        hidden_states, masks_queries_logits = self.transformer_module(multi_scale, mask_features)
        class_queries_logits = [self.class_predictor(h.transpose(0, 1)) for h in hidden_states]

        if self.training:
            return self._losses(masks_queries_logits, class_queries_logits, targets)
        return self._postprocess(masks_queries_logits[-1], class_queries_logits[-1], (H, W))

    # ------------------------------------------------------------------ train
    def _losses(self, masks_queries_logits, class_queries_logits, targets):
        mask_labels, class_labels = [], []
        for t in targets:
            masks = t["masks"]
            labels = t["labels"]
            # datasets generated before the get_target_from_mask fix dropped 1px-thick instances from
            # boxes/labels but not from masks, so the two can disagree in length; masks are the ground truth here
            if labels.shape[0] != masks.shape[0]:
                labels = torch.ones(masks.shape[0], dtype=torch.int64, device=masks.device)
            mask_labels.append(masks.float())
            class_labels.append(labels.long() - 1)  # foreground classes start at 0; num_labels is "no object"

        auxiliary = None
        if self.config.use_auxiliary_loss:
            auxiliary = [{"masks_queries_logits": m, "class_queries_logits": c}
                         for m, c in zip(masks_queries_logits[:-1], class_queries_logits[:-1])]
        loss_dict = self.criterion(masks_queries_logits[-1], class_queries_logits[-1], mask_labels, class_labels, auxiliary)

        # aggregate the final + auxiliary layers per term: the loop sums dict values and logs each key, so this
        # keeps the sum exact while giving three readable TensorBoard curves instead of 30
        grouped = OrderedDict((k, 0.0) for k in ("loss_cross_entropy", "loss_mask", "loss_dice"))
        for key, value in loss_dict.items():
            grouped[next(k for k in grouped if key.startswith(k))] += value
        self.last_loss_terms = {k: v.detach() for k, v in loss_dict.items()}
        return dict(grouped)

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def _postprocess(self, masks_queries_logits: Tensor, class_queries_logits: Tensor, image_size):
        """Standard Mask2Former instance inference: score = class prob x mean in-mask prob, top-k over queries x classes."""
        results = []
        num_labels = self.config.num_labels
        for mask_logits, class_logits in zip(masks_queries_logits, class_queries_logits):
            probs = class_logits.softmax(-1)[:, :num_labels]  # (Q, K), drops "no object"
            k = min(self.detections_per_img, probs.numel())
            cls_scores, flat_idx = probs.flatten().topk(k)
            query_idx = flat_idx // num_labels
            labels = flat_idx % num_labels + 1

            # final score <= class prob, so anything under the threshold here can't survive -- skip its upsample
            keep = cls_scores > self.score_thresh
            cls_scores, query_idx, labels = cls_scores[keep], query_idx[keep], labels[keep]
            if query_idx.numel() == 0:
                # interpolate rejects a 0-channel input, and an empty slice is routine (volume edges)
                results.append({
                    "masks": mask_logits.new_zeros((0, 1, *image_size)),
                    "scores": cls_scores, "labels": labels,
                    "boxes": mask_logits.new_zeros((0, 4)),
                })
                continue

            mask_probs = nn.functional.interpolate(
                mask_logits[query_idx][None], size=image_size, mode="bilinear", align_corners=False
            )[0].sigmoid()
            binary = mask_probs > 0.5
            mask_scores = (mask_probs * binary).flatten(1).sum(1) / (binary.flatten(1).sum(1) + 1e-6)
            scores = cls_scores * mask_scores

            keep = (scores > self.score_thresh) & binary.flatten(1).any(1)
            mask_probs, binary, scores, labels = mask_probs[keep], binary[keep], scores[keep], labels[keep]
            order = scores.argsort(descending=True)
            mask_probs, binary, scores, labels = mask_probs[order], binary[order], scores[order], labels[order]

            if self.overlap_thresh > 0 and binary.shape[0]:
                # panoptic-style: each pixel goes to the highest-scoring mask claiming it; a mask left with under
                # overlap_thresh of its own area was a duplicate of a better one and is dropped
                claimed = torch.zeros_like(binary[0])
                keep = torch.zeros(binary.shape[0], dtype=torch.bool, device=binary.device)
                for i in range(binary.shape[0]):
                    own = binary[i] & ~claimed
                    if own.sum() >= self.overlap_thresh * binary[i].sum():
                        keep[i] = True
                        claimed |= own
                        binary[i] = own
                mask_probs = mask_probs * binary
                mask_probs, binary, scores, labels = mask_probs[keep], binary[keep], scores[keep], labels[keep]

            boxes = masks_to_boxes(binary) if binary.shape[0] else torch.zeros((0, 4), device=mask_probs.device)
            results.append({"masks": mask_probs[:, None], "scores": scores, "labels": labels, "boxes": boxes})
        return results


HF_PREFIX_MAP = (
    ("model.pixel_level_module.decoder.", "pixel_decoder."),
    ("model.transformer_module.", "transformer_module."),
    ("class_predictor.", "class_predictor."),
)


def load_hf_head_weights(model: EMRMask2Former, source: str, logger=None):
    """Warm-start the pixel decoder / transformer decoder / class head from an HF Mask2Former checkpoint
    (hub id such as facebook/mask2former-swin-tiny-coco-instance, a snapshot dir, or a .safetensors file).
    The backbone is never loaded (HF's Swin differs from torchvision's), and tensors whose shapes don't match
    (class head, num_queries != 100, ResNet input projections) are skipped and reported."""
    import os
    from safetensors.torch import load_file

    if os.path.isfile(source):
        path = source
    elif os.path.isdir(source):
        path = os.path.join(source, "model.safetensors")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(source, "model.safetensors")
    hf_state = load_file(path)

    own_state = model.state_dict()
    loaded, skipped = {}, []
    for key, value in hf_state.items():
        for hf_prefix, own_prefix in HF_PREFIX_MAP:
            if key.startswith(hf_prefix):
                own_key = own_prefix + key[len(hf_prefix):]
                if own_key in own_state and own_state[own_key].shape == value.shape:
                    loaded[own_key] = value
                else:
                    skipped.append(own_key)
                break
    model.load_state_dict(loaded, strict=False)
    msg = f"Loaded {len(loaded)} Mask2Former head tensors from {source}; skipped {len(skipped)}: {skipped}"
    if logger is not None:
        logger.info(msg)
    return loaded, skipped
