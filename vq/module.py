from __future__ import annotations
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from . import activations
from .alias_free_torch import Activation1d, Activation1dWithCondition
from torch.nn.utils import weight_norm
from termcolor import colored

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def build_codec_activation(
    dim: int,
    activation_type: str = "SnakeBeta",
    leaky_relu_params: Optional[Dict[str, Any]] = None,
    speaker_condition: bool = False,
    condition_dim: int = 1024,
    f0_speaker_condition: bool = False,
    f0_condition_dim: int = 128,
    alpha_logscale: bool = True,
    no_condition: bool = False,
    log_context: Optional[str] = None,
    log_details: Optional[str] = None,
) -> nn.Module:
    if activation_type == "LeakyReLU":
        if not leaky_relu_params or "negative_slope" not in leaky_relu_params:
            raise ValueError("LeakyReLU activation requires leaky_relu_params['negative_slope'].")
        return nn.LeakyReLU(negative_slope=leaky_relu_params["negative_slope"])

    family = None
    if isinstance(activation_type, str):
        if activation_type.startswith("SnakeBeta"):
            family = "SnakeBeta"

    if family is None:
        raise ValueError(
            "Unsupported activation: "
            f"{activation_type}. Supported activations are 'SnakeBeta' and 'LeakyReLU'."
        )

    activation_name = family
    activation_kwargs: Dict[str, Any] = {
        "alpha_logscale": alpha_logscale,
    }
    if no_condition or not speaker_condition:
        activation_cls = getattr(activations, family)
        activation = Activation1d(activation=activation_cls(dim, **activation_kwargs))
    elif speaker_condition and not f0_speaker_condition:
        activation_name = f"{family}WithCondition"
        activation_cls = getattr(activations, activation_name)
        activation = Activation1dWithCondition(
            activation=activation_cls(dim, condition_dim, **activation_kwargs)
        )
    else:
        activation_name = f"{family}WithTimeVaryingCondition"
        activation_cls = getattr(activations, activation_name)
        activation = Activation1dWithCondition(
            activation=activation_cls(
                dim,
                condition_dim + f0_condition_dim,
                **activation_kwargs,
            )
        )

    if log_context is not None:
        details = f": {log_details}" if log_details else ""
        print(colored(f"{log_context} using {activation_name}{details}", "yellow"))

    return activation


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16,
                 dilation: int = 1,
                 speaker_condition=False,
                 f0_speaker_condition=False,
                 condition_dim=1024,
                 f0_condition_dim=128,
                 activation_type = 'SnakeBeta',
                 leaky_relu_params = None,
                 no_condition=False):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        activation = build_codec_activation(
            dim=dim,
            activation_type=activation_type,
            leaky_relu_params=leaky_relu_params,
            speaker_condition=speaker_condition,
            condition_dim=condition_dim,
            f0_speaker_condition=f0_speaker_condition,
            f0_condition_dim=f0_condition_dim,
            alpha_logscale=True,
            no_condition=no_condition,
            log_context=None if no_condition or activation_type == "LeakyReLU" else "ResidualUnit",
            log_details=f"dilation={dilation}",
        )

        self.block = nn.Sequential(
            activation,
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            activation,
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x, condition=None):
        res = x
        for i, layer in enumerate(self.block):
            if isinstance(layer, Activation1dWithCondition):
                x = layer(x, condition)
            else:
                x = layer(x)
        return x + res

class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1, dilations = (1, 3, 9), speaker_condition = False, condition_dim = 1024, activation_type = 'SnakeBeta', leaky_relu_params = None, input_dim: int = None):
        super().__init__()
        input_dim = dim // 2 if input_dim is None else input_dim
        runits = [ResidualUnit(input_dim, dilation=d, speaker_condition=speaker_condition, condition_dim=condition_dim, activation_type=activation_type, leaky_relu_params=leaky_relu_params) for d in dilations]
        activation = build_codec_activation(
            dim=input_dim,
            activation_type=activation_type,
            leaky_relu_params=leaky_relu_params,
            speaker_condition=speaker_condition,
            condition_dim=condition_dim,
            alpha_logscale=True,
        )
        # stride=1: use kernel_size=3, padding=1 to maintain temporal length
        # stride>1: use kernel_size=2*stride for downsampling
        if stride == 1:
            conv = WNConv1d(input_dim, dim, kernel_size=3, stride=1, padding=1)
        else:
            conv = WNConv1d(
                input_dim,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2 + stride % 2,
            )
        
        self.block = nn.Sequential(
            *runits,
            activation,
            conv,
        )

    def forward(self, x, condition=None):
        for layer in self.block:
            if isinstance(layer, Activation1dWithCondition) or isinstance(layer, ResidualUnit):
                x = layer(x, condition)
            else:
                x = layer(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1, dilations = (1, 3, 9), speaker_condition = False, condition_dim = 1024, f0_condition = False, f0_condition_dim: int = 128, f0_speaker_condition = False, activation_type = 'SnakeBeta', leaky_relu_params = None, use_split_condition_optimization: bool = True):
        super().__init__()
        self.f0_condition = f0_condition
        self.f0_speaker_condition = f0_speaker_condition
        self.speaker_condition = speaker_condition
        self.use_split_condition_optimization = bool(use_split_condition_optimization)
        self.use_condition_only = speaker_condition and not (self.f0_condition or self.f0_speaker_condition)
        self.f0_condition_dim = f0_condition_dim if f0_condition_dim is not None else 0
        activation = build_codec_activation(
            dim=input_dim,
            activation_type=activation_type,
            leaky_relu_params=leaky_relu_params,
            speaker_condition=speaker_condition,
            condition_dim=condition_dim,
            f0_speaker_condition=f0_speaker_condition,
            f0_condition_dim=self.f0_condition_dim,
            alpha_logscale=True,
            log_context=None if activation_type == "LeakyReLU" else "DecoderBlock",
            log_details=f"f0_condition={f0_condition}, f0_speaker_condition={f0_speaker_condition}",
        )
        # stride=1: use Conv1d to maintain temporal length (no upsampling)
        # stride>1: use ConvTranspose1d for upsampling
        if stride == 1:
            # Use regular Conv1d with kernel_size=3, padding=1 to maintain length
            self.block = nn.Sequential(
                activation,
                WNConv1d(input_dim, output_dim, kernel_size=3, padding=1)
            )
        else:
            self.block = nn.Sequential(
                activation,
                WNConvTranspose1d(
                    input_dim,
                    output_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=stride // 2 + stride % 2,
                    output_padding=stride % 2,
                )
            )
        self.block.extend([
            ResidualUnit(
                output_dim,
                dilation=d,
                speaker_condition=speaker_condition,
                f0_speaker_condition=False, # ResidualUnit does not support f0_speaker_condition !! only speaker info
                f0_condition_dim=self.f0_condition_dim,
                condition_dim=condition_dim,
                activation_type=activation_type,
                leaky_relu_params=leaky_relu_params,
            )
            for d in dilations
        ])
        
        print(colored(f"DecoderBlock: f0_condition={self.f0_condition}, f0_speaker_condition={self.f0_speaker_condition}", "yellow"))
        if self.f0_speaker_condition:
            if self.f0_condition:
                concat_dim = self.f0_condition_dim + condition_dim + input_dim
            else: concat_dim = condition_dim + input_dim
        else:
            if self.f0_condition: concat_dim = self.f0_condition_dim + input_dim
        if self.f0_condition or self.f0_speaker_condition:
            self.f0_cond_conv = WNConv1d(concat_dim, input_dim, kernel_size=7, padding=3)
        elif self.use_condition_only:
            print(colored(f"DecoderBlock using WNConv1d for condition only: condition_dim={condition_dim}, input_dim={input_dim}", "yellow"))
            self.condition_conv = WNConv1d(condition_dim + input_dim, input_dim, kernel_size=7, padding=3)

    def forward(self, x, condition=None, f0_cond=None):
        if self.f0_condition or self.f0_speaker_condition:
            if self.f0_condition:
                # Handle 2D tensor (B, T) -> (B, 1, T) for interpolate
                if f0_cond is not None and f0_cond.dim() == 2:
                    f0_cond = f0_cond.unsqueeze(1)  # (B, T) -> (B, 1, T)
                if f0_cond is not None and x.shape[-1] != f0_cond.shape[-1]:
                    # print(colored(f"Interpolating f0 condition from {f0_cond.shape} to match x {x.shape}", "yellow"))
                    f0_cond = torch.nn.functional.interpolate(f0_cond, size=x.shape[-1], mode='nearest')
                if f0_cond is not None:
                    assert x.shape[-1] == f0_cond.shape[-1], f"f0 shape {f0_cond.shape} does not match x shape {x.shape}"
            if self.f0_speaker_condition:
                if self.f0_condition:
                    condition_ = condition.unsqueeze(-1).expand(-1, -1, x.shape[-1])
                    x = torch.cat([x, condition_, f0_cond], dim=1)
                else:
                    # print('Concatenating speaker condition without f0 condition.')
                    condition_ = condition.unsqueeze(-1).expand(-1, -1, x.shape[-1])
                    x = torch.cat([x, condition_], dim=1)
            else:
                if self.f0_condition:
                    x = torch.cat([x, f0_cond], dim=1)
            x = self.f0_cond_conv(x)
        elif self.use_condition_only and condition is not None:
            condition_ = condition.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x = torch.cat([x, condition_], dim=1)
            x = self.condition_conv(x)

        def build_time_condition(seq_len):
            if not self.f0_speaker_condition:
                return condition
            if self.use_split_condition_optimization and self.f0_condition:
                return (condition, f0_cond)
            condition_time = condition.unsqueeze(-1).expand(-1, -1, seq_len)
            if self.f0_condition:
                condition_time = torch.cat([condition_time, f0_cond], dim=1)
            return condition_time

        for i, layer in enumerate(self.block):
            if isinstance(layer, Activation1dWithCondition):
                cond_for_layer = build_time_condition(x.shape[-1]) if self.f0_speaker_condition else condition
                x = layer(x, cond_for_layer)
            elif isinstance(layer, ResidualUnit):
                x = layer(x, condition)
            else:
                x = layer(x)
        return x
    
class ResLSTM(nn.Module):
    def __init__(self, dimension: int,
                 num_layers: int = 2,
                 bidirectional: bool = False,
                 skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension if not bidirectional else dimension // 2,
                            num_layers, batch_first=True,
                            bidirectional=bidirectional)

    def forward(self, x):
        """
        Args:
            x: [B, F, T]

        Returns:
            y: [B, F, T]
        """
        x = rearrange(x, "b f t -> b t f")
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = rearrange(y, "b t f -> b f t")
        return y


def _as_plain_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    except Exception:
        pass
    return dict(cfg) if isinstance(cfg, dict) else {}


def build_res_temporal(
    dimension: int,
    rnn_type: str,
    num_layers: int,
    bidirectional: bool,
    skip: bool = True,
) -> nn.Module:
    """Build the LSTM temporal path used by the open config."""
    t = (rnn_type or "lstm").lower()
    if t == "lstm":
        return ResLSTM(
            dimension,
            num_layers=num_layers,
            bidirectional=bidirectional,
            skip=skip,
        )
    raise ValueError(f"Unsupported temporal type: {rnn_type!r}. Expected 'lstm'.")


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross Attention module for conditioning decoder input with speaker embeddings.
    
    Args:
        query_dim: dimension of query (decoder hidden state)
        key_dim: dimension of key/value (speaker embedding)
        num_heads: number of attention heads
        dropout: dropout probability
    """
    def __init__(self, query_dim: int, key_dim: int, num_heads: int = 8, dropout: float = 0.1, use_sdpa: Optional[bool] = None):
        super().__init__()
        assert query_dim % num_heads == 0, f"query_dim {query_dim} must be divisible by num_heads {num_heads}"
        
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
        # Preserve the current default (SDPA) while allowing callers to opt back
        # into the legacy manual attention path for exact old-model alignment.
        self.use_sdpa = True if use_sdpa is None else bool(use_sdpa)
    
    def forward(self, query, key_value):
        """
        Args:
            query: [B, C, T] - decoder hidden state
            key_value: [B, C_kv, T_kv] - speaker embedding sequence
        
        Returns:
            output: [B, C, T] - attended output
        """
        B, C, T = query.shape
        _, _, T_kv = key_value.shape
        
        # Reshape to [B, T, C] for attention
        q = query.transpose(1, 2)  # [B, T, C]
        kv = key_value.transpose(1, 2)  # [B, T_kv, C_kv]
        
        # Project and reshape for multi-head attention
        q = self.q_proj(q).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = self.k_proj(kv).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T_kv, D]
        v = self.v_proj(kv).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T_kv, D]
        
        if self.use_sdpa:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, T_kv]
            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        out = self.out_proj(out)
        
        # Reshape back to [B, C, T]
        out = out.transpose(1, 2)  # [B, C, T]
        
        return out


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    Args:
        dim: input/output dimension
        hidden_dim: hidden layer dimension (typically 4 * dim)
        dropout: dropout probability
    """
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, T]
        Returns:
            output: [B, C, T]
        """
        # Transpose to [B, T, C] for linear layers
        x = x.transpose(1, 2)
        x = self.net(x)
        # Transpose back to [B, C, T]
        return x.transpose(1, 2)


class CrossAttentionBlock(nn.Module):
    """
    Complete Cross-Attention block with Pre-Normalization:
    1. Layer Norm -> Multi-Head Cross Attention -> Residual
    2. Layer Norm -> Feed-Forward Network -> Residual
    
    Args:
        query_dim: dimension of query (decoder hidden state)
        key_dim: dimension of key/value (speaker embedding)
        num_heads: number of attention heads
        ffn_hidden_dim: FFN hidden dimension (default: 4 * query_dim)
        dropout: dropout probability
    """
    def __init__(self, query_dim: int, key_dim: int, num_heads: int = 8, 
                 ffn_hidden_dim: int = None, dropout: float = 0.1, use_sdpa: Optional[bool] = None):
        super().__init__()
        
        self.query_dim = query_dim
        
        # Multi-Head Cross Attention
        self.cross_attn = MultiHeadCrossAttention(
            query_dim=query_dim,
            key_dim=key_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_sdpa=use_sdpa,
        )
        
        # Layer Norm for attention (works on channel dim in [B, C, T] format)
        self.norm1 = nn.LayerNorm(query_dim)
        
        # Feed-Forward Network
        self.ffn = FeedForwardNetwork(
            dim=query_dim,
            hidden_dim=ffn_hidden_dim,
            dropout=dropout
        )
        
        # Layer Norm for FFN
        self.norm2 = nn.LayerNorm(query_dim)
        
        print(colored(f"CrossAttentionBlock: query_dim={query_dim}, key_dim={key_dim}, num_heads={num_heads}, ffn_hidden={ffn_hidden_dim or 4*query_dim}", "cyan", attrs=['bold']))
    
    def forward(self, query, key_value):
        """
        Args:
            query: [B, C, T] - decoder hidden state
            key_value: [B, C_kv, T_kv] - speaker embedding sequence
        
        Returns:
            output: [B, C, T] - processed output
        """
        # Pre-Norm: Normalize -> Cross-Attention -> Residual
        # Layer norm: transpose to [B, T, C], normalize, transpose back
        normed = self.norm1(query.transpose(1, 2)).transpose(1, 2)
        attn_out = self.cross_attn(normed, key_value)
        # Residual connection
        x = query + attn_out
        
        # Pre-Norm: Normalize -> FFN -> Residual
        normed = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        ffn_out = self.ffn(normed)
        # Residual connection
        x = x + ffn_out
        
        return x
