"""
Mask2Former transformer module: learned queries refined by a masked-attention decoder that cross-attends to
the pixel decoder's multi-scale features, predicting one mask per query from the per-pixel `mask_features`.

Forked from transformers 5.14.1 `models/mask2former/modeling_mask2former.py`
(Mask2FormerAttention .. Mask2FormerTransformerModule). Module/attribute names are kept identical so an HF
checkpoint's `model.transformer_module.*` weights load onto `Mask2FormerTransformerModule` unchanged.
Changes vs. upstream: plain nn.Module / M2FConfig, no GradientCheckpointingLayer, no output_attentions /
output_hidden_states plumbing, returns plain tuples.

Late slice fusion does not change this file: EMRMask2Former collapses the per-slice `mask_features` before
calling Mask2FormerTransformerModule, so every decoder layer's mask prediction (and hence its attention mask)
is computed from the fused per-pixel embedding. A per-query variant would hook into Mask2FormerMaskPredictor.
"""
import torch
from torch import Tensor, nn

from emrmodel.mask2former.config import M2FConfig
from emrmodel.mask2former.pixel_decoder import Mask2FormerSinePositionEmbedding


class Mask2FormerAttention(nn.Module):
    """Multi-head self-attention with position embeddings added to queries and keys (DETR style)."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got {self.embed_dim} and {num_heads}).")
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: Tensor, position_embeddings: Tensor = None) -> Tensor:
        """hidden_states / position_embeddings: (seq_len, batch, embed_dim). Returns the same shape."""
        hidden_states = hidden_states.permute(1, 0, 2)
        position_embeddings = position_embeddings.permute(1, 0, 2) if position_embeddings is not None else None
        batch_size, target_len, embed_dim = hidden_states.size()

        hidden_states_original = hidden_states
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
        value_states = self._shape(self.v_proj(hidden_states_original), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim).transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)
        return self.out_proj(attn_output).permute(1, 0, 2)


class Mask2FormerMaskedAttentionDecoderLayer(nn.Module):
    """Masked cross-attention -> self-attention -> FFN (the cross/self order is swapped vs. a DETR decoder)."""

    def __init__(self, config: M2FConfig):
        super().__init__()
        self.embed_dim = config.hidden_dim
        self.pre_norm = config.pre_norm
        self.self_attn = Mask2FormerAttention(embed_dim=self.embed_dim, num_heads=config.num_attention_heads, dropout=config.dropout)
        self.dropout = config.dropout
        self.activation_fn = {"relu": nn.functional.relu, "gelu": nn.functional.gelu}[config.activation_function]
        self.activation_dropout = config.dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, config.num_attention_heads, config.dropout)
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.dim_feedforward)
        self.fc2 = nn.Linear(config.dim_feedforward, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def _cross_attn(self, hidden_states, level_index, position_embeddings, query_position_embeddings, encoder_hidden_states, encoder_attention_mask):
        return self.cross_attn(
            query=self.with_pos_embed(hidden_states, query_position_embeddings),
            key=self.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]),
            value=encoder_hidden_states[level_index],
            attn_mask=encoder_attention_mask,
            key_padding_mask=None,
        )[0]

    def _drop(self, x, p):
        return nn.functional.dropout(x, p=p, training=self.training)

    def forward_post(self, hidden_states, level_index, position_embeddings, query_position_embeddings,
                     encoder_hidden_states, encoder_attention_mask):
        residual = hidden_states
        hidden_states = self._cross_attn(hidden_states, level_index, position_embeddings, query_position_embeddings,
                                         encoder_hidden_states, encoder_attention_mask)
        hidden_states = self.cross_attn_layer_norm(residual + self._drop(hidden_states, self.dropout))

        residual = hidden_states
        hidden_states = self.self_attn(hidden_states=hidden_states, position_embeddings=query_position_embeddings)
        hidden_states = self.self_attn_layer_norm(residual + self._drop(hidden_states, self.dropout))

        residual = hidden_states
        hidden_states = self._drop(self.activation_fn(self.fc1(hidden_states)), self.activation_dropout)
        hidden_states = self._drop(self.fc2(hidden_states), self.dropout)
        return self.final_layer_norm(residual + hidden_states)

    def forward_pre(self, hidden_states, level_index, position_embeddings, query_position_embeddings,
                    encoder_hidden_states, encoder_attention_mask):
        residual = hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)
        hidden_states = self._cross_attn(hidden_states, level_index, position_embeddings, query_position_embeddings,
                                         encoder_hidden_states, encoder_attention_mask)
        hidden_states = residual + self._drop(hidden_states, self.dropout)

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, position_embeddings=query_position_embeddings)
        hidden_states = residual + self._drop(hidden_states, self.dropout)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self._drop(self.activation_fn(self.fc1(hidden_states)), self.activation_dropout)
        hidden_states = self._drop(self.fc2(hidden_states), self.dropout)
        return residual + hidden_states

    def forward(self, hidden_states, level_index, position_embeddings, query_position_embeddings,
                encoder_hidden_states, encoder_attention_mask):
        """
        hidden_states / query_position_embeddings: (num_queries, B, C); encoder_hidden_states /
        position_embeddings: per-level lists of (h*w, B, C); encoder_attention_mask: (B*heads, Q, h*w) bool.
        """
        fn = self.forward_pre if self.pre_norm else self.forward_post
        return fn(hidden_states, level_index, position_embeddings, query_position_embeddings,
                  encoder_hidden_states, encoder_attention_mask)


class Mask2FormerMaskedAttentionDecoder(nn.Module):
    def __init__(self, config: M2FConfig):
        super().__init__()
        self.mask_feature_size = config.mask_feature_size
        self.layerdrop = config.dropout  # upstream reuses dropout as layerdrop; 0 by default
        self.num_feature_levels = 3
        self.decoder_layers = config.decoder_layers - 1
        self.layers = nn.ModuleList([Mask2FormerMaskedAttentionDecoderLayer(config) for _ in range(self.decoder_layers)])
        self.layernorm = nn.LayerNorm(config.hidden_dim)
        self.mask_predictor = Mask2FormerMaskPredictor(
            hidden_size=config.hidden_dim, num_heads=config.num_attention_heads, mask_feature_size=self.mask_feature_size
        )

    def forward(self, inputs_embeds, multi_stage_positional_embeddings, pixel_embeddings, encoder_hidden_states,
                query_position_embeddings, feature_size_list):
        """
        Returns (intermediate_hidden_states, intermediate_mask_predictions): one entry for the query init plus one
        per decoder layer; each hidden state is (num_queries, B, C), each mask prediction (B, Q, H/4, W/4).
        """
        hidden_states = inputs_embeds
        intermediate_hidden_states = self.layernorm(inputs_embeds)
        intermediate = (intermediate_hidden_states,)
        predicted_mask, attention_mask = self.mask_predictor(intermediate_hidden_states, pixel_embeddings, feature_size_list[0])
        intermediate_mask_predictions = (predicted_mask,)

        for idx, decoder_layer in enumerate(self.layers):
            if self.training and torch.rand([]) < self.layerdrop:
                continue

            level_index = idx % self.num_feature_levels
            # a query whose mask is empty at this level would attend to nothing -> let it attend everywhere
            where = (attention_mask.sum(-1) != attention_mask.shape[-1]).to(attention_mask.dtype)
            attention_mask = attention_mask * where.unsqueeze(-1)

            hidden_states = decoder_layer(
                hidden_states,
                level_index,
                multi_stage_positional_embeddings,
                query_position_embeddings,
                encoder_hidden_states,
                encoder_attention_mask=attention_mask,
            )

            intermediate_hidden_states = self.layernorm(hidden_states)
            predicted_mask, attention_mask = self.mask_predictor(
                intermediate_hidden_states, pixel_embeddings, feature_size_list[(idx + 1) % self.num_feature_levels]
            )
            intermediate_mask_predictions += (predicted_mask,)
            intermediate += (intermediate_hidden_states,)

        return intermediate, intermediate_mask_predictions


class Mask2FormerPredictionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module) -> None:
        super().__init__()
        self.layers = [nn.Linear(in_dim, out_dim), activation]
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class Mask2FormerMLPPredictionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            activation = nn.ReLU() if i < num_layers - 1 else nn.Identity()
            layer = Mask2FormerPredictionBlock(in_dim, out_dim, activation=activation)
            self.layers.append(layer)
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class Mask2FormerMaskPredictor(nn.Module):
    """Mask logits = <mask embedding of query, per-pixel embedding>; also the binarized attention mask for the next layer."""

    def __init__(self, hidden_size: int, num_heads: int, mask_feature_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mask_embedder = Mask2FormerMLPPredictionHead(self.hidden_size, self.hidden_size, mask_feature_size)

    def forward(self, outputs: Tensor, pixel_embeddings: Tensor, attention_mask_target_size=None):
        mask_embeddings = self.mask_embedder(outputs.transpose(0, 1))
        outputs_mask = torch.einsum("bqc, bchw -> bqhw", mask_embeddings, pixel_embeddings)

        attention_mask = nn.functional.interpolate(outputs_mask, size=attention_mask_target_size, mode="bilinear", align_corners=False)
        attention_mask = attention_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attention_mask = (attention_mask.flatten(0, 1) < 0.5).bool().detach()
        return outputs_mask, attention_mask


class Mask2FormerTransformerModule(nn.Module):
    def __init__(self, in_features: int, config: M2FConfig):
        super().__init__()
        hidden_dim = config.hidden_dim
        self.num_feature_levels = 3
        self.position_embedder = Mask2FormerSinePositionEmbedding(num_position_features=hidden_dim // 2, normalize=True)
        self.queries_embedder = nn.Embedding(config.num_queries, hidden_dim)
        self.queries_features = nn.Embedding(config.num_queries, hidden_dim)
        # plain list, as upstream: identity unless channels differ (they don't with the defaults), so no weights
        self.input_projections = []
        for _ in range(self.num_feature_levels):
            if in_features != hidden_dim or config.enforce_input_projection:
                self.input_projections.append(nn.Conv2d(in_features, hidden_dim, kernel_size=1))
            else:
                self.input_projections.append(nn.Sequential())
        self.decoder = Mask2FormerMaskedAttentionDecoder(config=config)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

    def forward(self, multi_scale_features, mask_features: Tensor):
        multi_stage_features = []
        multi_stage_positional_embeddings = []
        size_list = []

        for i in range(self.num_feature_levels):
            feature = multi_scale_features[i]
            size_list.append(feature.shape[-2:])
            pos = self.position_embedder(feature.shape, feature.device, feature.dtype, None).flatten(2)
            multi_stage_positional_embeddings.append(pos.permute(2, 0, 1))
            proj = self.input_projections[i](feature).flatten(2) + self.level_embed.weight[i][None, :, None]
            multi_stage_features.append(proj.permute(2, 0, 1))

        _, batch_size, _ = multi_stage_features[0].shape
        query_embeddings = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_features = self.queries_features.weight.unsqueeze(1).repeat(1, batch_size, 1)

        return self.decoder(
            inputs_embeds=query_features,
            multi_stage_positional_embeddings=multi_stage_positional_embeddings,
            pixel_embeddings=mask_features,
            encoder_hidden_states=multi_stage_features,
            query_position_embeddings=query_embeddings,
            feature_size_list=size_list,
        )
