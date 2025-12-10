from torchvision.models.detection.backbone_utils import BackboneWithFPN, _validate_trainable_layers, _resnet_fpn_extractor
from torchvision.models.resnet import resnet50, ResNet50_Weights, ResNet
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, LastLevelMaxPool


from typing import Optional, List, Callable, OrderedDict
import torch
from torch import nn


class Stacked_Resnet50FPN_Backbone(BackboneWithFPN):
    def __init__(self, num_slices):
        self.num_slices = num_slices

        # Proxy backbone for accessing attributes like out_channels in_channels etc.
        norm_layer = misc_nn_ops.FrozenBatchNorm2d
        trainable_backbone_layers = _validate_trainable_layers(True, None, max_value=5, default_value=3) # trainable backbone layers is passed as None
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, progress=True, norm_layer=norm_layer)
        fpn_maker_tuple = _resnet_fpn_tuple_maker(backbone=backbone, trainable_layers=trainable_backbone_layers, norm_layer=norm_layer)
        super().__init__(fpn_maker_tuple[0], fpn_maker_tuple[1], fpn_maker_tuple[2], fpn_maker_tuple[3])
        self.body.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor(B, C, H, W)): input batch of C number of 2d slices

        Returns:
            OrderedDict: The FPN results for each stage of the backbone - '0', '1', '2', '3', 'pool'. Each of these keys returns a list of feature maps, where each item in the list is an individual slice processed by each seperate backbone.
        """
        stacked_features = OrderedDict()
        for slice_id in range(self.num_slices):
            slice = x[:, slice_id, :, :] # (B, H, W)
            slice = slice.unsqueeze(1)  # (B, 1, H, W)
            slice_features = self.fpn(self.body(slice))
            for key in slice_features.keys():
                feature_map = slice_features.get(key) # feature map 
                level_features = stacked_features.get(key, [])
                level_features.append(feature_map)
                stacked_features.update({key: level_features})
        return stacked_features # OrderedDict { '0': list(featureMap(slice1),featureMap(slice2), featureMap(slice3)}



def _resnet_fpn_tuple_maker(
    backbone: ResNet,
    trainable_layers: int,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
) -> tuple:

    # select layers that won't be frozen
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return (backbone, return_layers, in_channels_list, out_channels, extra_blocks, norm_layer)
