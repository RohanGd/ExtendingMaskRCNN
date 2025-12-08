import torch
import torch.nn as nn
import torch.nn.functional as F

class SliceSEFusion(nn.Module):
    """
    Squeeze-and-Excitation style fusion over slices.

    Inputs:
        feats_per_slice: list of length S
            each element is a tensor of shape [B, C, H, W]

    Output:
        fused: tensor of shape [B, C, H, W]
    """
    def __init__(self, num_slices: int, channels: int, reduction: int = 16):
        super().__init__()
        self.num_slices = num_slices
        self.channels = channels

        hidden = max(channels // reduction, 4)

        # Small MLP that maps per-slice channel summary -> 1 scalar logit
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, 1)

        # Optional static bias per slice (helps encode "center slice is usually best")
        self.static_logits = nn.Parameter(torch.zeros(num_slices))

    def forward(self, feats_per_slice):
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
        g_flat = g.reshape(B * self.num_slices, C)

        # MLP to get one logit per slice element: [B*S, 1] -> [B, S]
        h = F.relu(self.fc1(g_flat))
        logits_dynamic = self.fc2(h).reshape(B, self.num_slices)

        # Add static per-slice logits and softmax along slice dimension
        logits = logits_dynamic + self.static_logits  # broadcast over batch
        weights = F.softmax(logits, dim=1)            # [B, S]

        # Apply weights to feature maps: [B, S, 1, 1, 1] for broadcasting
        w = weights.view(B, self.num_slices, 1, 1, 1)
        fused = (x * w).sum(dim=1)  # [B, C, H, W]

        return fused

