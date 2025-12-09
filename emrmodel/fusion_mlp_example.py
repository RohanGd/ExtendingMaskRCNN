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

class SliceSEFusionFixedWindow(nn.Module):
    """
    Squeeze-and-Excitation style fusion over slices.

    Inputs:
        feats_per_slice: list of length S
            each element is a tensor of shape [B, C, H, W]

    Output:
        fused: tensor of shape [B, C, H, W]
    """
    def __init__(self, num_slices: int, channels: int, reduction: int = 16, window_size=7):
        super().__init__()
        self.num_slices = num_slices
        self.channels = channels
        self.k = window_size

        hidden = max(channels // reduction, 4)

        # Small MLP that maps per-slice channel summary -> 1 scalar logit
        self.fc1 = nn.Linear(channels, hidden) # 256 * 16 = 1536 + 2560 = 4096
        self.fc2 = nn.Linear(hidden, 1) # 16 * 1 = 16

        # Optional static bias per slice (helps encode "center slice is usually best")
        self.static_logits = nn.Parameter(torch.zeros(num_slices))

    def forward(self, feats_per_slice):
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

        x = F.pad(x, (0, pad_w, 0, pad_h))
        print(x.shape)

        x = x.view(B, S, C, k*k, (H + pad_h)//k, (W + pad_w)//k) # B, S , C, K*K, ~ H / K, ~ W /K

        print(x.shape)
        # pooling over K*k -> B, S, C, K*K
        g = x.mean(dim=(-1, -2))
        g = g.view(B, S, k*k, C) 
        print(g.shape)

        g_flat = g.reshape(B*S*k*k, C)
        
        # MLP to get one logit per slice element: [B*S, 1] -> [B, S]
        h = F.relu(self.fc1(g_flat))
        print(h.shape)
        logits_dynamic = self.fc2(h).reshape(B, k*k, self.num_slices)
        print(logits_dynamic.shape)

        # # Add static per-slice logits and softmax along slice dimension
        logits = logits_dynamic + self.static_logits  # broadcast over batch
        weights = F.softmax(logits, dim=1)            # [B, K*K, S]

        print(weights.shape)

        # Apply weights to feature maps: [B, S, 1, 1, 1] for broadcasting
        w = weights.view(B, self.num_slices, 1, k*k, 1, 1)

        print(w.shape, x.shape)

        fused = (x* w).sum(dim=1)  # [B, C, k*k, ~H/k, ~W/k]
        h_k, w_k = fused.shape[-2], fused.shape[-1]
        fused = fused.reshape(B, C, k * h_k, k * w_k)
        print(fused.shape)
        return fused

    





if __name__ == "__main__":
    obj = SliceSEFusionFixedWindow(3, 256)

    x = torch.rand((2, 256, 161, 163))
    # x = torch.rand((2, 256, 21, 21))

    x = list((x, x, x)) # 3, 2, 256, 10, 10  S, B, C, H, W

    obj(x)