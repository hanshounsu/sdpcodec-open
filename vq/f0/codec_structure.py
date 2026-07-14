"""
F0 Codec using CodecEncoder/CodecDecoder structure.
This is an alternative to the Jukebox-style F0Encoder/F0Decoder.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.amp import autocast
from torch.nn.utils.weight_norm import _weight_norm
from ..module import WNConv1d, DecoderBlock, EncoderBlock, CrossAttentionBlock, build_res_temporal
from termcolor import colored


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        if m.bias is not None:
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)


def _tensor_debug_stats(x: Optional[torch.Tensor]):
    if x is None:
        return None
    with torch.no_grad():
        x_detached = x.detach().float()
        finite_mask = torch.isfinite(x_detached)
        stats = {
            "shape": tuple(x_detached.shape),
            "non_finite": int((~finite_mask).sum().item()),
        }
        if finite_mask.any():
            finite_vals = x_detached[finite_mask]
            stats.update(
                {
                    "min": float(finite_vals.min().item()),
                    "max": float(finite_vals.max().item()),
                    "mean": float(finite_vals.mean().item()),
                    "absmax": float(finite_vals.abs().max().item()),
                }
            )
        return stats


def normalize_fcpe_loss_mode(fcpe_loss_mode=None, legacy_use_fcpe_loss=None) -> str:
    if fcpe_loss_mode is None:
        return 'dense' if bool(legacy_use_fcpe_loss) else 'none'
    mode = str(fcpe_loss_mode).strip().lower()
    if mode in {'', 'none', 'off', 'false', '0', 'disabled'}:
        return 'none'
    if mode in {'true', '1', 'on', 'enabled', 'bce', 'wbce'}:
        return 'dense'
    if mode == 'dense':
        return mode
    raise ValueError(f"Unsupported fcpe_loss_mode: {fcpe_loss_mode}")


def fcpe_loss_mode_enabled(fcpe_loss_mode: str) -> bool:
    return normalize_fcpe_loss_mode(fcpe_loss_mode) != 'none'


def build_channel_schedule(ngf, num_levels, max_channels=None):
    channels = []
    current = int(ngf)
    max_channels = None if max_channels is None else int(max_channels)
    for _ in range(int(num_levels)):
        current = current * 2
        if max_channels is not None:
            current = min(current, max_channels)
        channels.append(int(current))
    return channels


class ProjectedSpeakerConcatFuser(nn.Module):
    """Exact 1x1 speaker concat path without expanding global speaker features over time."""

    def __init__(self, input_dim: int, condition_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.condition_proj = nn.Linear(condition_dim, self.input_dim)
        self.merge_conv = WNConv1d(self.input_dim * 2, self.input_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        projected = self.condition_proj(condition)
        # Use .weight so this works whether weight_norm is active (the hook recomputes
        # it from weight_v/weight_g on access) or has been removed for inference
        # (folded plain weight). Accessing weight_v directly crashes after removal.
        weight = self.merge_conv.weight
        bias = self.merge_conv.bias
        x_out = F.conv1d(x, weight[:, :self.input_dim, :], bias=bias)
        cond_out = F.linear(projected, weight[:, self.input_dim:, 0]).unsqueeze(-1)
        return x_out + cond_out


class F0CodecEncoder(nn.Module):
    """
    F0 Encoder using CodecEncoder structure.
    Converts F0 input (1 or 2 channels) to latent representation.
    """
    def __init__(self,
                 input_channels=1,  # 1 for f0 only, 2 for f0+vuv
                 ngf=16,
                 max_channels=None,
                 use_rnn=True,
                 rnn_bidirectional=False,
                 rnn_num_layers=2,
                 rnn_type='lstm',
                 up_ratios=(3, 4, 5, 8),
                 dilations=(1, 3, 9),
                 out_channels=128,
                 activation_type='SnakeBeta',
                 leaky_relu_params=None,
                 ):
        super().__init__()
        if activation_type != 'LeakyReLU':
            raise ValueError("F0CodecEncoder supports only activation_type='LeakyReLU'.")
        if leaky_relu_params is None:
            leaky_relu_params = {'negative_slope': 0.1}
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios
        self.input_channels = input_channels
        self.max_channels = None if max_channels is None else int(max_channels)
        self.stage_channels = build_channel_schedule(ngf, len(up_ratios), max_channels=self.max_channels)
        self.temporal_layer_index = None
        
        # Create first convolution (adapt to input_channels)
        prev_dim = int(ngf)
        self.block = [WNConv1d(input_channels, prev_dim, kernel_size=7, padding=3)]
        
        # Create EncoderBlocks with progressive channel growth capped by max_channels.
        for stage_dim, stride in zip(self.stage_channels, up_ratios):
            self.block += [EncoderBlock(stage_dim, stride=stride, dilations=dilations,
                                       speaker_condition=False,  # F0 doesn't use speaker condition
                                       activation_type=activation_type, 
                                       leaky_relu_params=leaky_relu_params,
                                       input_dim=prev_dim)]
            prev_dim = stage_dim
        d_model = prev_dim
        
        # RNN
        if use_rnn:
            temporal_layer = build_res_temporal(
                d_model,
                rnn_type,
                num_layers=rnn_num_layers,
                bidirectional=rnn_bidirectional,
            )
            self.block.append(temporal_layer)
            self.temporal_layer_index = len(self.block) - 1
        
        # Create last convolution
        activation = nn.LeakyReLU(negative_slope=leaky_relu_params['negative_slope'])
        
        self.block += [
            activation,
            WNConv1d(d_model, out_channels, kernel_size=3, padding=1),
        ]
        
        # Wrap block into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model
        
        # Build width_list for compatibility with F0Decoder
        self.width_list = self._build_width_list()
        
        self.reset_parameters()
        
        print(f"F0CodecEncoder: input_channels={input_channels}, out_channels={out_channels}, "
              f"hop_length={self.hop_length}, ngf={ngf}, up_ratios={up_ratios}, max_channels={self.max_channels}")
        print(f"F0CodecEncoder stage_channels: {self.stage_channels}")
        if self.temporal_layer_index is not None:
            print(
                f"F0CodecEncoder temporal: type={rnn_type}, "
                f"num_layers={rnn_num_layers}, bidirectional={rnn_bidirectional}"
            )
        print(f"Number of parameters in F0CodecEncoder: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")
    
    def _build_width_list(self):
        """
        Build width_list for compatibility with F0Decoder that expects it.
        Should have num_levels items (one per EncoderBlock), NOT num_levels + 1.
        """
        return [list(self.stage_channels)]
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T) where C is 1 or 2 (f0 or f0+vuv)
        Returns:
            (B, out_channels, T')
        """
        # Match F0Encoder behavior: if input_channels==1, use only first channel
        # This allows receiving 2-channel input but only using f0 channel
        if self.input_channels == 1:
            x = x[:, :1, :]  # use only f0 for encoding
        
        return self.block(x)
    
    def reset_parameters(self):
        self.apply(init_weights)


class F0CodecDecoder(nn.Module):
    """
    F0 Decoder using CodecDecoder structure (without quantizer).
    Converts latent representation back to F0 output (1 or 2 channels).
    """
    def __init__(self,
                 in_channels=128,
                 upsample_initial_channel=None,  # Auto-calculated if None
                 output_channels=1,  # 1 for f0 only, 2 for f0+vuv
                 ngf=16,
                 max_channels=None,
                 use_rnn=True,
                 rnn_bidirectional=False,
                 rnn_num_layers=2,
                 rnn_type='lstm',
                 up_ratios=(8, 5, 4, 3),
                 dilations=(1, 3, 9),
                 activation_type='SnakeBeta',
                 leaky_relu_params=None,
                 fcpe_out_dims=360,  # FCPE latent dimension
                 fcpe_loss_mode=None,
                 use_fcpe_loss=None,  # Deprecated compatibility shim; prefer fcpe_loss_mode.
                 speaker_condition=False,  # Use speaker embedding as condition
                 condition_dim=0,  # Speaker embedding dimension
                 use_spk_concat=True,
                 use_spk_film=False,
                 use_mhca=False,
                 mhca_num_heads=2,
                 mhca_dropout=0.1,
                 mhca_key_dim=128,
                 mhca_use_sdpa: Optional[bool] = None,
                 use_split_condition_optimization: bool = True,
                 ):
        super().__init__()
        if activation_type != 'LeakyReLU':
            raise ValueError("F0CodecDecoder supports only activation_type='LeakyReLU'.")
        if leaky_relu_params is None:
            leaky_relu_params = {'negative_slope': 0.1}
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.fcpe_out_dims = fcpe_out_dims
        self.fcpe_loss_mode = normalize_fcpe_loss_mode(fcpe_loss_mode, use_fcpe_loss)
        self.fcpe_loss_enabled = fcpe_loss_mode_enabled(self.fcpe_loss_mode)
        self.use_fcpe_loss = self.fcpe_loss_enabled
        self.use_spk_concat = bool(use_spk_concat)
        self.use_spk_film = bool(use_spk_film)
        self.use_split_condition_optimization = bool(use_split_condition_optimization)
        self.speaker_condition = bool(speaker_condition) and (self.use_spk_concat or self.use_spk_film)
        self.condition_dim = condition_dim
        self.use_mhca = use_mhca
        self.mhca_key_dim = mhca_key_dim
        self.max_channels = None if max_channels is None else int(max_channels)
        self.stage_channels = build_channel_schedule(ngf, len(up_ratios), max_channels=self.max_channels)
        self.temporal_layer_index = None
        
        # Auto-calculate upsample_initial_channel to match encoder's final channel
        if upsample_initial_channel is None:
            upsample_initial_channel = self.stage_channels[-1]
        self.upsample_initial_channel = upsample_initial_channel
        self.decoder_output_channels = list(reversed([int(ngf)] + list(self.stage_channels[:-1])))
        self.decoder_input_channels = [self.upsample_initial_channel] + self.decoder_output_channels[:-1]

        channels = self.upsample_initial_channel
        # First conv: no speaker conditioning here (will be added before each DecoderBlock)
        layers = [WNConv1d(in_channels, channels, kernel_size=7, padding=3)]
        
        self.spk_concat_fusers = nn.ModuleList() if self.speaker_condition and self.use_spk_concat else None
        self.spk_proj_layers = None
        self.spk_merge_layers = None
        if self.spk_concat_fusers is not None:
            if self.use_split_condition_optimization:
                for block_input_dim in self.decoder_input_channels:
                    self.spk_concat_fusers.append(
                        ProjectedSpeakerConcatFuser(
                            input_dim=block_input_dim,
                            condition_dim=condition_dim,
                        )
                    )
            else:
                self.spk_proj_layers = nn.ModuleList()
                self.spk_merge_layers = nn.ModuleList()
                for block_input_dim in self.decoder_input_channels:
                    self.spk_proj_layers.append(
                        nn.Linear(condition_dim, block_input_dim)
                    )
                    self.spk_merge_layers.append(
                        WNConv1d(block_input_dim * 2, block_input_dim, kernel_size=1)
                    )
                    self.spk_concat_fusers.append(nn.Identity())

        self.spk_film_layers = nn.ModuleList() if self.speaker_condition and self.use_spk_film else None
        if self.spk_film_layers is not None:
            for block_input_dim in self.decoder_input_channels:
                self.spk_film_layers.append(
                    nn.Linear(condition_dim, block_input_dim * 2)
                )
        
        # RNN
        if use_rnn:
            temporal_layer = build_res_temporal(
                channels,
                rnn_type,
                num_layers=rnn_num_layers,
                bidirectional=rnn_bidirectional,
            )
            layers.append(temporal_layer)
            self.temporal_layer_index = len(layers) - 1
        
        # Decoder blocks
        for input_dim, output_dim, stride in zip(self.decoder_input_channels, self.decoder_output_channels, up_ratios):
            layers += [DecoderBlock(input_dim, output_dim, stride, dilations, 
                                   speaker_condition=False,  # F0 doesn't use speaker condition
                                   f0_condition=False,  # F0 decoder doesn't condition on F0
                                   activation_type=activation_type,
                                   leaky_relu_params=leaky_relu_params,
                                   )]
        
        # Final activation and output
        # Skip final scalar-F0 layers when the decoder predicts an FCPE distribution.
        if not self.fcpe_loss_enabled:
            activation = nn.LeakyReLU(negative_slope=leaky_relu_params['negative_slope'])
            
            layers += [
                activation,
                WNConv1d(output_dim, output_channels, kernel_size=7, padding=3),
            ]
        
        # Don't use Sequential - we need to access intermediate outputs
        self.layers = nn.ModuleList(layers)
        
        # FCPE-style latent prediction head (optional)
        # Keep logits in fp32 for a more stable BCEWithLogits path.
        if self.fcpe_loss_enabled:
            self.fcpe_head = WNConv1d(output_dim, self.fcpe_out_dims, kernel_size=1)
            self.last_fcpe_logits = None
            self.last_forward_debug = {}
            print(
                "F0CodecDecoder: FCPE loss enabled "
                f"(mode={self.fcpe_loss_mode}, latent_bins={self.fcpe_out_dims})"
            )
        else:
            self.last_forward_debug = {}
        
        # Cross-Attention Blocks (optional)
        if self.use_mhca:
            self.mhca_list = nn.ModuleList()
            for input_dim in self.decoder_input_channels:
                self.mhca_list.append(
                    CrossAttentionBlock(
                        query_dim=input_dim,
                        key_dim=mhca_key_dim,
                        num_heads=mhca_num_heads,
                        ffn_hidden_dim=None,
                        dropout=mhca_dropout,
                        use_sdpa=mhca_use_sdpa,
                    )
                )
            print(colored(f"F0CodecDecoder: Using Cross-Attention Blocks (key_dim={mhca_key_dim}, heads={mhca_num_heads})", "cyan"))
        else:
            self.mhca_list = None

        # Store indices of DecoderBlocks for later use
        self.decoder_block_indices = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DecoderBlock):
                self.decoder_block_indices.append(i)
        
        # Build width_list for compatibility with CodecDecoder's f0_width_list
        # This represents the channel dimensions at each decoder level
        self.width_list = self._build_width_list()
        
        self.reset_parameters()
        
        print(f"F0CodecDecoder: in_channels={in_channels}, output_channels={output_channels}, "
              f"hop_length={self.hop_length}, ngf={ngf}, upsample_initial_channel={self.upsample_initial_channel}, up_ratios={up_ratios}, max_channels={self.max_channels}")
        print(f"F0CodecDecoder decoder_input_channels: {self.decoder_input_channels}")
        print(f"F0CodecDecoder decoder_output_channels: {self.decoder_output_channels}")
        print(f"F0CodecDecoder width_list: {self.width_list}")
        if self.speaker_condition:
            print(f"F0CodecDecoder: Speaker conditioning enabled (condition_dim={condition_dim})")
            if self.use_spk_concat:
                print(f"  → Speaker concat enabled before each DecoderBlock input ({len(up_ratios)} blocks)")
            if self.use_spk_film:
                print(f"  → Speaker FiLM enabled before each DecoderBlock input ({len(up_ratios)} blocks)")
        if self.temporal_layer_index is not None:
            print(
                f"F0CodecDecoder temporal: type={rnn_type}, "
                f"num_layers={rnn_num_layers}, bidirectional={rnn_bidirectional}"
            )
        print(f"Number of parameters in F0CodecDecoder: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")
    
    def _build_width_list(self):
        """
        Build width_list compatible with CodecDecoder conditioning.
        reversed(width_list[0]) should match:
        [after_rnn] + [decoder block outputs except the last block]
        """
        valid_stage_dims = [self.upsample_initial_channel] + self.decoder_output_channels[:-1]
        return [list(reversed(valid_stage_dims))]
    
    def forward(self, x, spk_emb=None):
        """
        Args:
            x: (B, in_channels, T')
            spk_emb: (B, condition_dim) speaker embedding (optional, required if speaker_condition=True)
                     If use_mhca=True, this can be a tuple (spk_emb, mhca_key) where
                     mhca_key is a time-varying sequence for cross-attention (e.g., x_quantized).
        Returns:
            if self.fcpe_loss_enabled:
                tuple: (outs, fcpe_latent)
                - outs: list of outputs for F0 conditioning
                - fcpe_latent: (B, fcpe_out_dims, T) probabilities for FCPE decoding/monitoring
            else:
                outs: list of outputs for F0 conditioning
        """
        # Split speaker embedding and MHCA key if tuple provided
        if self.use_mhca and isinstance(spk_emb, tuple):
            spk_emb_global, mhca_key = spk_emb
        else:
            spk_emb_global = spk_emb
            mhca_key = None
        
        # Validate speaker embedding if conditioning is enabled
        if self.speaker_condition:
            assert spk_emb_global is not None, "Speaker embedding is required when speaker_condition=True"
        
        # Match original F0Decoder format from codec.py line 185-197
        outs = []
        debug_info = {
            "input": _tensor_debug_stats(x),
            "speaker": _tensor_debug_stats(spk_emb_global) if isinstance(spk_emb_global, torch.Tensor) else None,
            "mhca_key": _tensor_debug_stats(mhca_key) if isinstance(mhca_key, torch.Tensor) else None,
        }
        
        decoder_block_count = 0
        mhca_idx = 0
        
        # Process through all layers
        for i, layer in enumerate(self.layers):
            if self.temporal_layer_index is not None and i == self.temporal_layer_index:
                # After the temporal block, append to outs (this becomes outs[0]).
                x = layer(x)
                debug_info["after_temporal"] = _tensor_debug_stats(x)
                outs.append(x)
            elif isinstance(layer, DecoderBlock):
                # Optional MHCA before each DecoderBlock
                if self.use_mhca and mhca_key is not None:
                    x = self.mhca_list[mhca_idx](x, mhca_key)
                    debug_info[f"after_mhca_{mhca_idx}"] = _tensor_debug_stats(x)
                    mhca_idx += 1

                if self.speaker_condition and self.use_spk_film:
                    gamma_beta = self.spk_film_layers[decoder_block_count](spk_emb_global)
                    gamma, beta = gamma_beta.chunk(2, dim=1)
                    gamma = gamma.unsqueeze(-1)
                    beta = beta.unsqueeze(-1)
                    x = x * (1.0 + gamma) + beta

                # Concatenate speaker embedding BEFORE each DecoderBlock
                if self.speaker_condition and self.use_spk_concat:
                    if self.use_split_condition_optimization:
                        x = self.spk_concat_fusers[decoder_block_count](x, spk_emb_global)
                    else:
                        # Project speaker embedding: (B, condition_dim) -> (B, block_input_dim)
                        spk_emb_proj = self.spk_proj_layers[decoder_block_count](spk_emb_global)  # (B, block_input_dim)
                        # Broadcast to match temporal dimension: (B, block_input_dim) -> (B, block_input_dim, T)
                        spk_emb_broadcast = spk_emb_proj.unsqueeze(-1).expand(-1, -1, x.shape[-1])
                        # Concatenate: (B, block_input_dim, T) + (B, block_input_dim, T) -> (B, 2*block_input_dim, T)
                        x_with_spk = torch.cat([x, spk_emb_broadcast], dim=1)
                        # Merge back to block_input_dim using 1x1 conv
                        x = self.spk_merge_layers[decoder_block_count](x_with_spk)  # (B, block_input_dim, T)
                
                decoder_block_count += 1
                # DecoderBlock.block = [activation, WNConvTranspose1d (index 1), ResidualUnit, ...]
                # Collect output after the stride-changing conv. Skip the final block output
                # to match the original conditioning convention used by CodecDecoder.
                for j, sub_layer in enumerate(layer.block):
                    x = sub_layer(x)
                    if j == 1 and decoder_block_count < len(self.up_ratios):
                        debug_info[f"after_decoder_block_{decoder_block_count}_stride"] = _tensor_debug_stats(x)
                        outs.append(x)
            else:
                # Regular layers (WNConv1d, Activation, etc.)
                x = layer(x)
        
        # Skip one index (to match original structure at index 4)
        outs.append(None)  # outs[4]: placeholder (not used)
        
        # Handle final output based on fcpe_loss_mode.
        if self.fcpe_loss_enabled:
            # x is now the last DecoderBlock output (B, 16, T)
            # Run the FCPE head in fp32 and expose dense logits for BCEWithLogits.
            debug_info["pre_fcpe"] = _tensor_debug_stats(x)
            with autocast(device_type=x.device.type, enabled=False):
                fcpe_logits = self.fcpe_head(x.float())
                fcpe_latent = torch.sigmoid(fcpe_logits)
            self.last_fcpe_logits = fcpe_logits
            debug_info["fcpe_logits"] = _tensor_debug_stats(fcpe_logits)
            self.last_forward_debug = debug_info
            # For audio conditioning, we still need some F0 output
            # Use a dummy placeholder (won't be used for loss, only for conditioning)
            outs.append(None)  # outs[5]: placeholder (FCPE mode doesn't need final F0)
            return outs, fcpe_latent
        else:
            self.last_fcpe_logits = None
            debug_info["final_output"] = _tensor_debug_stats(x)
            self.last_forward_debug = debug_info
            # Normal mode: x is the final F0 reconstruction after activation + conv
            outs.append(x)  # outs[5]: final F0 reconstruction
            return outs

    def reset_parameters(self):
        self.apply(init_weights)
