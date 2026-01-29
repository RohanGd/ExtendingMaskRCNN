"""
OPtion1: SliceSEFusion
For the fpn_level i, for i in '0', '1', '2', '3', '4', 'pool':

we get a list(len=num_slices) of B,C,H,W which is the output of the resnet50 where each item in the list is the output of resnet50fpnbackbone for one slice at ith feature level.

stack the list to get [B,S,C,H,W] Tensor. 
Now for normal SliceSEFusion we did global average pooling to simply get one value per channel, then pass it through the MLP to get [B, S]. 
Scale the orignal stacked tensor by the B, S; i.e. scale each slice after weighing it through MLP per batch.

 Option 2:
Same thing but based on pre-defined window sizes. Basically, divide feature map into e.g. 14x14 windows, compute Squeeze and Excitation weights for each local area, and reassemble back to the full map.

Say we have [B=2, S=3, C=256, H=28, W=28] torch.unfold will give [B, S, 2, 2, C, 14, 14]

APPROACH 1: ALTHOUGH THIS IS ACTUALLy [B, S, C, 2, 2, C, 14, 14] assume it to be, [4, B, S, C, 14, 14] for conceptual simplicity.
Simply pass each B, S, C, 14, 14 to the SLiceSEFusion to get [4, B, C, 14, 14]
    what this means is that for one [B, S, C, 14, 14] I will have done:
        B, S, C -> mean along C for that 14*14 window.
        then B, S -> MLP to weight each Slice for that 14*14 window.
        then reweight each slice in that B, S, C, 14*14 and sum to return B, S, 14, 14.
    So I will have 4 : B, C, 14, 14 which I can concatenate to get [B, C, 28, 28]

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from emrConfigManager import Fusion_Logger

class IdentityFusion(nn.Module):
    def __init__(self, num_slices: None, channels: None, reduction: int = 16, init_bias=None):
        super().__init__()
        self.static_logits = torch.zeros((0,))
        return
    def forward(self, feats_per_slice, *args):
        return feats_per_slice


class SliceSEFusion(nn.Module):
    """
    Squeeze-and-Excitation style fusion over slices.

    Inputs:
        feats_per_slice: list of length S
            each element is a tensor of shape [B, C, H, W]

    Output:
        fused: tensor of shape [B, C, H, W]
    """
    def __init__(self, num_slices: int, channels: int, reduction: int = 16, init_bias=None):
        super().__init__()
        self.num_slices = num_slices
        self.channels = channels

        self.hidden = max(channels // reduction, 4)

        # Small MLP that maps per-slice channel summary -> 1 scalar logit
        self.mlp = nn.Sequential(
            nn.Linear(channels, self.hidden), # params: 256 * 16 = 1536 + 2560 = 4096
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, 1) # params: 16 * 1 = 16
        )

        # Optional static bias per slice (helps encode "center slice is usually best")
        # self.static_logits = self.get_init_bias(init_bias_type=init_bias)

    def get_init_bias(self, init_bias_type=None):
        if init_bias_type == None or init_bias_type == "zero":
            bias = torch.zeros(self.num_slices)
        elif init_bias_type == "only_center":
            bias = [0.0 for _ in range(self.num_slices)]
            bias[self.num_slices//2] = 1
            bias = torch.tensor(bias)
        elif init_bias_type == "gaussian":
            x = torch.linspace(-1, 1, self.num_slices)
            bias = torch.exp(-x**2 / (2*5))  # sigma = 5
            bias /= bias.sum()
        return nn.Parameter(bias)
    
    def forward(self, feats_per_slice, *args):
        """
        feats_per_slice: list of length S, each [B, C, H, W]
        """
        assert len(feats_per_slice) == self.num_slices
        B, C, H, W = feats_per_slice[0].shape

        # Stack slices: [B, S, C, H, W]
        x = torch.stack(feats_per_slice, dim=1)

        # Global average pooling over H, W -> [B, S, C]
        g = x.mean(dim=(-1, -2))

        # Flatten slice dimension into batch for MLP: [B*S, C]
        # MLP to get one logit per slice element: [B*S, 1] -> [B, S]
        logits_dynamic = self.mlp(g).squeeze(-1)
        # Add static per-slice logits and softmax along slice dimension
        # logits = logits_dynamic + self.static_logits  # broadcast over batch
        weights = F.softmax(logits_dynamic, dim=1)            # [B, S]

        Fusion_Logger.log(logits_dynamic.detach(), self.static_logits.detach(), weights.detach(), weights_shape=weights.shape)
        # Apply weights to feature maps: [B, S, 1, 1, 1] for broadcasting
        w = weights.view(B, self.num_slices, 1, 1, 1)
        fused = (x * w).sum(dim=1)  # [B, C, H, W]

        return fused

class SliceSEFusionFixedWindow(SliceSEFusion):
    """
    Squeeze-and-Excitation style fusion over slices.

    Inputs:
        feats_per_slice: list of length S
            each element is a tensor of shape [B, C, H, W]

    Output:
        fused: tensor of shape [B, C, H, W]
    """
    def __init__(self, num_slices: int, channels: int, reduction: int = 16, window_size=1, init_bias=None):
        super().__init__(num_slices, channels, reduction, init_bias)
        self.k = window_size

    def forward(self, feats_per_slice, *args):
        """
        feats_per_slice: list of length S, each [B, C, H, W]
        """
        assert len(feats_per_slice) == self.num_slices

        # Stack slices: [B, S, C, H, W]
        x = torch.stack(feats_per_slice, dim=1)
        B, S, C, H, W = x.shape
        k = self.k
        pad_h = (k - H % k) % k
        pad_w = (k - W % k) % k
        x = F.pad(x, (0, pad_w, 0, pad_h))  # padding is added only to the bottom and right
        Hp, Wp = H + pad_h, W + pad_w

        x = x.unfold(3, k, k).unfold(4, k, k) # [B, S, C, ~H/k, ~W/k, k, k] 
        # mean pooling over K*k -> [B, S, C, ~H/k, ~W/k]
        
        g = x.mean(dim=(-1, -2)) # [B, S, C, ~H/k, ~W/k]
        g = g.permute(0,1,3,4,2) # [B, S, H/k, W/k, C]

        # MLP to get one logit per slice element: [B, ~H/k, ~W/k, S, C] -> [B, ~H/k, ~W/k, S, 1]
        logits_dynamic = self.mlp(g)
        logits_dynamic = logits_dynamic.squeeze(-1)

        weights = F.softmax(logits_dynamic, dim=-1)            # [B, ~H/k, ~W/k, S]
        Fusion_Logger.log(logits_dynamic.detach(), self.static_logits.detach(), weights.detach(), weights_shape=weights.shape)
        weights = weights[:, :, None, :, :, None, None] # [B, S, C, ~H/k, ~W/k, k, k]

        fused = (x * weights).sum(1) # [B, C, ~H/k, ~W/k, k, k]
        fused = fused.permute(0, 1, 4, 2, 5, 3)
        fused = fused.reshape(B, C, Hp, Wp)
        fused = fused[:, :, :H, :W]

        return fused


class SlicePixelAttention(SliceSEFusion):
    def __init__(self, num_slices, channels, reduction = 16, init_bias=None):
        super().__init__(num_slices, channels, reduction, init_bias)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, self.hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden, num_slices, kernel_size=1)
        )

    def forward(self, feats_per_slice, *args):
        x = torch.stack(feats_per_slice, dim=1)  # [B, S, C, H, W]

        x_mean = x.mean(dim=1)  # [B, C, H, W]
        logits = self.mlp(x_mean)  # [B, S, H, W]
        logits = logits + self.static_logits[:, None, None]
        weights = torch.softmax(logits, dim=1)  # [B, S, H, W]
        Fusion_Logger.log(logits.detach(), self.static_logits.detach(), weights.detach(), weights_shape=weights.shape)

        fused = (x * weights[:, :, None, :, :]).sum(dim=1)
        return fused


class SliceSEFusionPerFPN(nn.Module):
    def __init__(self, num_slices, channels, reduction = 16, init_bias=None, fusionType=None):
        super().__init__()

        if fusionType == None:
            self.fusionClassType = SliceSEFusion
        else:
            self.fusionClassType = fusionType

        self.fusionModules = nn.ModuleDict({
            '0': self.fusionClassType(num_slices=num_slices, channels=channels, reduction=reduction, init_bias="only_center"),
            '1': self.fusionClassType(num_slices=num_slices, channels=channels, reduction=reduction, init_bias="gaussian"),
            '2': self.fusionClassType(num_slices=num_slices, channels=channels, reduction=reduction, init_bias="gaussian"),
            '3': self.fusionClassType(num_slices=num_slices, channels=channels, reduction=reduction, init_bias="zero"),
            'pool': self.fusionClassType(num_slices=num_slices, channels=channels, reduction=reduction, init_bias="zero")
        }) # hardcoding init_bias and resnet50fpn levels
    
    def forward(self, feats_per_slice, key, *args):
        x = self.fusionModules[key].forward(feats_per_slice)
        return x

class SlicePixelAttentionPerFPN(SliceSEFusionPerFPN):
    def __init__(self, num_slices, channels, reduction = 16, init_bias=None):
        super().__init__(num_slices, channels, reduction, init_bias, fusionType=SlicePixelAttention)
