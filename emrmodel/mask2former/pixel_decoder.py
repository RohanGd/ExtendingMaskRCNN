"""
Mask2Former pixel decoder: multi-scale deformable-attention encoder over the stride 8/16/32 backbone maps, plus
an FPN-style top-down path to the stride-4 per-pixel `mask_features`.

Forked from transformers 5.14.1 `models/mask2former/modeling_mask2former.py`
(Mask2FormerSinePositionEmbedding .. Mask2FormerPixelDecoder). Module/attribute names are kept identical
so an HF checkpoint's `model.pixel_level_module.decoder.*` weights load onto `Mask2FormerPixelDecoder`
with only the prefix rewritten. Changes vs. upstream:
  - plain nn.Module / M2FConfig, ModelOutput dataclasses replaced by plain tuples, no output_attentions /
    output_hidden_states plumbing, no torch.compile lru_cache on the position embedding;
  - EARLY FUSION HOOK (`slice_fusion` argument of Mask2FormerPixelDecoder.forward): every backbone level may
    be a list of S per-slice maps instead of one tensor. Each slice is projected to `feature_size` channels
    by that level's own 1x1 projection (`input_projections[*]` / `adapter_1`), and the S projected maps are
    collapsed by `slice_fusion(list_of_S, level_key)` -- the same call signature emrmodel/early_mlp_fusion.py's
    modules have in emrmodel/generalized_rcnn.py. This is the counterpart of fusing FPN outputs before the RPN.
"""
import math
import warnings

import numpy as np
import torch
from torch import Tensor, nn

from emrmodel.mask2former.config import M2FConfig


class Mask2FormerSinePositionEmbedding(nn.Module):
    """DETR-style 2D sine position embedding."""

    def __init__(self, num_position_features: int = 64, temperature: int = 10000, normalize: bool = False, scale: float = None):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_position_features = num_position_features
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, shape: torch.Size, device, dtype, mask: Tensor = None) -> Tensor:
        batch_size, _, height, width = shape
        if mask is None:
            y_embed = torch.arange(1, height + 1, dtype=dtype, device=device)[None, :, None].expand(batch_size, height, width)
            x_embed = torch.arange(1, width + 1, dtype=dtype, device=device)[None, None, :].expand(batch_size, height, width)
        else:
            embed_mask = mask.to(dtype)
            y_embed = embed_mask.cumsum(1)
            x_embed = embed_mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_position_features, dtype=torch.int64, device=device).to(dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_position_features)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


def multi_scale_deformable_attention(value: Tensor, value_spatial_shapes, sampling_locations: Tensor, attention_weights: Tensor) -> Tensor:
    """Pure-PyTorch multi-scale deformable attention (grid_sample per level), no CUDA extension needed."""
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([height * width for height, width in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # -> (B*heads, head_dim, H, W)
        value_l_ = value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        # -> (B*heads, queries, points, 2)
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = nn.functional.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(batch_size * num_heads, 1, num_queries, num_levels * num_points)
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention(nn.Module):
    """Multiscale deformable attention as proposed in Deformable DETR."""

    def __init__(self, embed_dim: int, num_heads: int, n_levels: int, n_points: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim (d_model) must be divisible by num_heads, but got {embed_dim} and {num_heads}")
        dim_per_head = embed_dim // num_heads
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn("Make embed_dim / num_heads a power of 2 for the efficient deformable attention path.")

        self.im2col_step = 128
        self.d_model = embed_dim
        self.n_levels = n_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * n_levels * n_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, position_embeddings=None,
                reference_points=None, spatial_shapes_list=None):
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        if sum(height * width for height, width in spatial_shapes_list) != sequence_length:
            raise ValueError("Make sure to align the spatial shapes with the sequence length of the encoder hidden states")

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = value.masked_fill(attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = nn.functional.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(
                [[shape[1], shape[0]] for shape in spatial_shapes_list], dtype=torch.long, device=reference_points.device
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        output = multi_scale_deformable_attention(value, spatial_shapes_list, sampling_locations, attention_weights)
        return self.output_proj(output)


class Mask2FormerPixelDecoderEncoderLayer(nn.Module):
    def __init__(self, config: M2FConfig):
        super().__init__()
        self.embed_dim = config.feature_size
        self.self_attn = Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention(
            embed_dim=self.embed_dim, num_heads=config.num_attention_heads, n_levels=3, n_points=4
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = nn.functional.relu
        self.activation_dropout = config.dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_feedforward_dim)
        self.fc2 = nn.Linear(config.encoder_feedforward_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states, attention_mask, position_embeddings=None, reference_points=None, spatial_shapes_list=None):
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes_list=spatial_shapes_list,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.training and not torch.isfinite(hidden_states).all():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states


class Mask2FormerPixelDecoderEncoderOnly(nn.Module):
    """Stack of `encoder_layers` deformable-attention layers over the flattened multi-scale maps."""

    def __init__(self, config: M2FConfig):
        super().__init__()
        self.layers = nn.ModuleList([Mask2FormerPixelDecoderEncoderLayer(config) for _ in range(config.encoder_layers)])

    @staticmethod
    def get_reference_points(spatial_shapes_list, valid_ratios, device):
        reference_points_list = []
        for lvl, (height, width) in enumerate(spatial_shapes_list):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=valid_ratios.dtype, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=valid_ratios.dtype, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * width)
            reference_points_list.append(torch.stack((ref_x, ref_y), -1))
        reference_points = torch.cat(reference_points_list, 1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def forward(self, inputs_embeds, attention_mask, position_embeddings, spatial_shapes_list, valid_ratios):
        hidden_states = inputs_embeds
        reference_points = self.get_reference_points(spatial_shapes_list, valid_ratios, device=inputs_embeds.device)
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                position_embeddings=position_embeddings,
                reference_points=reference_points,
                spatial_shapes_list=spatial_shapes_list,
            )
        return hidden_states


class Mask2FormerPixelDecoder(nn.Module):
    def __init__(self, config: M2FConfig, feature_channels):
        """
        feature_channels: backbone channels per level, ordered high -> low resolution (strides 4, 8, 16, 32).
        """
        super().__init__()
        feature_dim = config.feature_size
        mask_dim = config.mask_feature_size
        self.position_embedding = Mask2FormerSinePositionEmbedding(num_position_features=feature_dim // 2, normalize=True)
        self.num_feature_levels = 3
        transformer_in_channels = feature_channels[-self.num_feature_levels:]
        self.transformer_feature_strides = config.feature_strides[-self.num_feature_levels:]
        self.feature_channels = feature_channels
        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, feature_dim))

        # 1x1 projections for the encoder levels, ordered low -> high resolution (stride 32, 16, 8)
        self.input_projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, feature_dim, kernel_size=1), nn.GroupNorm(32, feature_dim))
            for in_channels in transformer_in_channels[::-1]
        ])

        self.encoder = Mask2FormerPixelDecoderEncoderOnly(config)
        self.mask_projection = nn.Conv2d(feature_dim, mask_dim, kernel_size=1, stride=1, padding=0)

        # extra FPN levels below the encoder (just stride 4 with the default strides)
        stride = min(self.transformer_feature_strides)
        self.common_stride = config.common_stride
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []
        for idx, in_channels in enumerate(self.feature_channels[: self.num_fpn_levels]):
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channels, feature_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, feature_dim),
            )
            output_conv = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(32, feature_dim),
                nn.ReLU(),
            )
            self.add_module(f"adapter_{idx + 1}", lateral_conv)
            self.add_module(f"layer_{idx + 1}", output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # low -> high resolution
        self.lateral_convolutions = lateral_convs[::-1]
        self.output_convolutions = output_convs[::-1]

    @staticmethod
    def get_valid_ratio(mask, dtype=torch.float32):
        _, height, width = mask.shape
        valid_height = torch.sum(~mask[:, :, 0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        return torch.stack([valid_width.to(dtype) / width, valid_height.to(dtype) / height], -1)

    @staticmethod
    def _project(projection: nn.Module, feature, level_key: str, slice_fusion):
        """Project one backbone level; for a per-slice list, project each slice then fuse across slices."""
        if isinstance(feature, (list, tuple)):
            return slice_fusion([projection(f) for f in feature], level_key)
        return projection(feature)

    def forward(self, features, slice_fusion=None):
        """
        features: 4 backbone levels ordered high -> low resolution (strides 4, 8, 16, 32), each either a tensor
            (B, C_l, H_l, W_l), or -- for early fusion -- a list of S such per-slice tensors, in which case
            `slice_fusion(list_of_S_projected_maps, level_key)` collapses them after projection. level_key is the
            backbone level name '0'..'3' (stride 4..32).
        Returns (mask_features (B, mask_dim, H/4, W/4), multi_scale_features: 3 maps, low -> high resolution).
        """
        num_levels = len(features)
        level_keys = [str(i) for i in range(num_levels)]

        inputs_embeds = []
        position_embeddings = []
        for level, (x, key) in enumerate(zip(features[::-1][: self.num_feature_levels], level_keys[::-1])):
            embed = self._project(self.input_projections[level], x, key, slice_fusion)
            inputs_embeds.append(embed)
            position_embeddings.append(self.position_embedding(embed.shape, embed.device, embed.dtype))

        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in inputs_embeds]

        # flatten for the encoder
        spatial_shapes_list = [(embed.shape[2], embed.shape[3]) for embed in inputs_embeds]
        input_embeds_flat = torch.cat([embed.flatten(2).transpose(1, 2) for embed in inputs_embeds], 1)
        masks_flat = torch.cat([mask.flatten(1) for mask in masks], 1)
        position_embeddings = [embed.flatten(2).transpose(1, 2) for embed in position_embeddings]
        level_pos_embed_flat = torch.cat(
            [x + self.level_embed[i].view(1, 1, -1) for i, x in enumerate(position_embeddings)], 1
        )
        valid_ratios = torch.stack([self.get_valid_ratio(mask, dtype=input_embeds_flat.dtype) for mask in masks], 1)

        last_hidden_state = self.encoder(
            inputs_embeds=input_embeds_flat,
            attention_mask=masks_flat,
            position_embeddings=level_pos_embed_flat,
            spatial_shapes_list=spatial_shapes_list,
            valid_ratios=valid_ratios,
        )
        batch_size = last_hidden_state.shape[0]

        split_sizes = [height * width for height, width in spatial_shapes_list]
        encoder_output = torch.split(last_hidden_state, split_sizes, dim=1)
        outputs = [
            x.transpose(1, 2).view(batch_size, -1, spatial_shapes_list[i][0], spatial_shapes_list[i][1])
            for i, x in enumerate(encoder_output)
        ]

        # extra FPN levels, low -> high resolution
        for idx, (feature, key) in enumerate(zip(features[: self.num_fpn_levels][::-1], level_keys[: self.num_fpn_levels][::-1])):
            current_fpn = self._project(self.lateral_convolutions[idx], feature, key, slice_fusion)
            out = current_fpn + nn.functional.interpolate(
                outputs[-1], size=current_fpn.shape[-2:], mode="bilinear", align_corners=False
            )
            outputs.append(self.output_convolutions[idx](out))

        multi_scale_features = tuple(outputs[: self.num_feature_levels])
        return self.mask_projection(outputs[-1]), multi_scale_features
