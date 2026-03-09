
import torch
from torch import nn

class MeanMaskFeaturesFusion(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, mask_features):
        """(num_proposals_across_batch, 256, S, H, W) -> collapse slice dim by taking mean"""
        return torch.mean(mask_features, dim=2)

class ChooseCenterMaskFeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, mask_features):
        center_slice = mask_features.shape[2] // 2
        return mask_features[:, :, center_slice, :, :]

class Conv3dMaskFeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 1, 1))

    def forward(self, mask_features):
        fused = self.fusion(mask_features)
        return fused.squeeze(2)
    
class SEMaskFeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, mask_features):
        # N, C, S, ,H, W
        inp_feat:torch.Tensor = mask_features.mean(dim=(-1, -2)) # N, C, S
        inp_feat = inp_feat.permute(0, 2, 1) # N, S, C

        weights = self.mlp(inp_feat) # N, S, 1
        weights = weights.permute(0, 2, 1) # N, 1, S

        weights = weights[:, :, :, None, None]
        weights = torch.softmax(weights, dim=2)
        fused = (mask_features * weights).sum(dim=2)

        return fused



def get_mask_features_fusion(mask_features_fusion:str="only_center"):
    print("CALLED", mask_features_fusion)
    if mask_features_fusion == "mean":
        return MeanMaskFeaturesFusion()
    if mask_features_fusion == "only_center":
        return ChooseCenterMaskFeatureFusion()
    if mask_features_fusion == "conv3d":
        return Conv3dMaskFeatureFusion()
    if mask_features_fusion == "SE":
        return SEMaskFeatureFusion()