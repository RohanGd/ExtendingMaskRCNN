"""
Plain-dataclass replacement for transformers' Mask2FormerConfig, carrying only the fields the forked
modules in this package read. Defaults match Mask2FormerConfig (transformers 5.14.1), so an HF
checkpoint's head weights line up with a model built from M2FConfig().
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class M2FConfig:
    num_labels: int = 1                      # foreground classes; "no object" is added on top
    feature_size: int = 256                  # pixel decoder width
    mask_feature_size: int = 256
    hidden_dim: int = 256                    # transformer decoder width
    encoder_feedforward_dim: int = 1024
    activation_function: str = "relu"
    encoder_layers: int = 6                  # deformable-attention layers in the pixel decoder
    decoder_layers: int = 10                 # masked-attention decoder runs decoder_layers - 1 layers
    num_attention_heads: int = 8
    dropout: float = 0.0
    dim_feedforward: int = 2048
    pre_norm: bool = False
    enforce_input_projection: bool = False
    common_stride: int = 4
    num_queries: int = 100
    no_object_weight: float = 0.1
    class_weight: float = 2.0
    mask_weight: float = 5.0
    dice_weight: float = 5.0
    train_num_points: int = 12544
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75
    init_std: float = 0.02
    init_xavier_std: float = 1.0
    use_auxiliary_loss: bool = True
    feature_strides: Tuple[int, ...] = field(default=(4, 8, 16, 32))
