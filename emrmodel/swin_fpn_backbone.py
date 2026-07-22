"""
Swin-T + FPN backbone, as an alternative to the resnet50-FPN backbones in stacked_fpn_backbone.py.

torchvision's swin_t().features is a Sequential of
    [patch_embed, stage1, merge, stage2, merge, stage3, merge, stage4]
(indices 0..7). Stage outputs sit at indices 1, 3, 5, 7 with channels [96, 192, 384, 768] and strides
[4, 8, 16, 32] relative to the input -- the same four strides torchvision's resnet50 FPN extractor
returns via layer1..layer4, so they slot into a FeaturePyramidNetwork the same way. Swin stages are
channel-last ((B, H, W, C)); each stage output is permuted to (B, C, H, W) before entering the FPN.
"""

from collections import OrderedDict
from typing import Optional

import torch
from torch import nn
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool, ExtraFPNBlock

SWIN_T_STAGE_CHANNELS = [96, 192, 384, 768]
SWIN_T_STAGE_FEATURE_INDICES = [1, 3, 5, 7]


def _build_swin_t_body(in_channels: int) -> nn.Module:
    """torchvision's swin_t feature extractor (patch embed + 4 stages, no final norm/head),
    with the patch-embed conv's input channels swapped for `in_channels`."""
    swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    body = swin.features
    old_conv = body[0][0]
    body[0][0] = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride)
    return body


class SwinStageExtractor(nn.Module):
    """Runs a swin_t `features` Sequential and returns the four per-stage feature maps as an
    OrderedDict keyed '0'..'3' (lowest to highest level), each (B, C, H, W) -- matching the keys
    torchvision's resnet extractor uses, so it slots into the same FeaturePyramidNetwork usage."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.body = _build_swin_t_body(in_channels)

    def forward(self, x):
        outputs = OrderedDict()
        stage = 0
        for i, module in enumerate(self.body):
            x = module(x)
            if i in SWIN_T_STAGE_FEATURE_INDICES:
                outputs[str(stage)] = x.permute(0, 3, 1, 2).contiguous()
                stage += 1
        return outputs


class SwinFPNBackbone(nn.Module):
    """Swin-T + FPN, drop-in replacement for torchvision's resnet50-FPN extractor. Used as the
    plain, single shared-weight backbone for the "channel fusion" path, where all n input slices
    are already stacked into the channel dimension before the backbone runs."""

    def __init__(self, in_channels: int, out_channels: int = 256, extra_blocks: Optional[ExtraFPNBlock] = None):
        super().__init__()
        self.body = SwinStageExtractor(in_channels)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=SWIN_T_STAGE_CHANNELS,
            out_channels=out_channels,
            extra_blocks=extra_blocks or LastLevelMaxPool(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        return self.fpn(self.body(x))


class Stacked_SwinFPN_Backbone(SwinFPNBackbone):
    """Each slice is passed through a single-channel Swin-T+FPN backbone (shared weights across
    slices, mirroring Stacked_Resnet50FPN_Backbone), and the resulting per-slice FPN features are
    kept as lists instead of fused, for use with early_mlp_fusion / roi_heads_fusion."""

    def __init__(self, num_slices: int, out_channels: int = 256):
        super().__init__(in_channels=1, out_channels=out_channels)
        self.num_slices = num_slices

    def forward(self, x):
        """
        Args:
            x (torch.Tensor(B, S, H, W)): batch of S consecutive 2d slices

        Returns:
            OrderedDict: FPN results per level ('0', '1', '2', '3', 'pool'), each a list of
            per-slice feature maps, mirroring Stacked_Resnet50FPN_Backbone's output.
        """
        stacked_features = OrderedDict()
        for slice_id in range(self.num_slices):
            slice_ = x[:, slice_id, :, :].unsqueeze(1)  # (B, 1, H, W)
            slice_features = super().forward(slice_)
            for key, feature_map in slice_features.items():
                stacked_features.setdefault(key, []).append(feature_map)
        return stacked_features
