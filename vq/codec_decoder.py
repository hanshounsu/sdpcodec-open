import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from .vq.residual_vq import ResidualVQ
from .module import WNConv1d, DecoderBlock, CrossAttentionBlock, build_codec_activation
from .temporal_config import CodecDecoderSpeakerConditionConfig, CodecDecoderTemporalConfig
from .alias_free_torch import Activation1dWithCondition
from termcolor import colored

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        if isinstance(getattr(m, "weight", None), nn.parameter.UninitializedParameter):
            return
        if m.bias is not None:
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

class CodecDecoder(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 upsample_initial_channel=1536,
                 ngf=48,
                 temporal: Optional[CodecDecoderTemporalConfig] = None,
                 up_ratios=(5, 5, 2, 2, 2),
                 dilations=(1, 3, 9),
                 quantizer_type="rvq",
                 vq_num_quantizers=1,
                 vq_dim=1024,
                 vq_commit_weight=0.25,
                 vq_weight_init=False,
                 vq_full_commit_loss=False,
                 quantizer_force_fp32: bool = False,
                 codebook_size=8192,
                 codebook_dim=8,
                 speaker_condition=False,
                 condition_dim=1024,
                 activation_type='SnakeBeta',
                 leaky_relu_params=None,
                 snake_logscale=True,
                 f0_condition=False,
                 f0_start_layer=0,
                 f0_end_layer=None,
                 f0_every=1,
                 f0_width_list=None,
                 f0_speaker_condition=False,
                 use_stage_speaker_film=True,
                 use_mhca=False,
                 spk_cond_use_concat=False,
                 mhca_num_heads=8,
                 mhca_dropout=0.1,
                 mhca_key_dim=128,
                 mhca_use_sdpa: Optional[bool] = None,
                 mhca_start_layer=0,
                 mhca_end_layer=None,
                 mhca_every=1,
                 spk_cond_start_layer=None,
                 spk_cond_end_layer=None,
                 spk_cond_every=None,
                 decoder_type='default',
                 use_split_condition_optimization: bool = True,
                ):
        super().__init__()
        self.temporal = temporal or CodecDecoderTemporalConfig()

        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios
        self.f0_condition = bool(f0_condition)
        self.f0_start_layer = int(f0_start_layer)
        self.f0_end_layer = f0_end_layer
        self.f0_every = max(1, int(f0_every))
        self.f0_width_list = []
        self.f0_block_enabled = [False] * len(up_ratios)
        self.spk_start_layer = int(mhca_start_layer if spk_cond_start_layer is None else spk_cond_start_layer)
        self.spk_end_layer = mhca_end_layer if spk_cond_end_layer is None else spk_cond_end_layer
        self.spk_every = max(1, int(mhca_every if spk_cond_every is None else spk_cond_every))
        self.spk_block_enabled = [False] * len(up_ratios)
        self.spk_concat_block_enabled = [False] * len(up_ratios)
        self.spk_film_block_enabled = [False] * len(up_ratios)
        self.use_mhca = use_mhca
        self.spk_cond_use_concat = bool(spk_cond_use_concat)
        self.use_stage_speaker_film = bool(use_stage_speaker_film)
        self.mhca_key_dim = mhca_key_dim
        self.mhca_start_layer = mhca_start_layer
        self.mhca_end_layer = mhca_end_layer
        self.mhca_every = max(1, int(mhca_every))
        self.decoder_type = str(decoder_type).lower()
        self.use_split_condition_optimization = bool(use_split_condition_optimization)

        if quantizer_type != "rvq":
            raise ValueError(
                "This open SDPCodec config surface supports only "
                "codec_decoder.quantizer_type='rvq'."
            )
        if quantizer_force_fp32:
            print(colored("CodecDecoder RVQ quantizer runs under fp32 autocast-disabled mode", "yellow", attrs=['bold']))
        self.quantizer = ResidualVQ(
            num_quantizers=vq_num_quantizers,
            dim=vq_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            commitment=vq_commit_weight,
            force_quantization_f32=quantizer_force_fp32,
            weight_init=vq_weight_init,
            full_commit_loss=vq_full_commit_loss,
        )
            
        total_decoder_blocks = len(self.up_ratios)
        if speaker_condition:
            self.spk_start_layer = max(0, min(int(self.spk_start_layer), total_decoder_blocks))
            if self.spk_end_layer is None:
                self.spk_end_layer = total_decoder_blocks
            else:
                self.spk_end_layer = max(self.spk_start_layer, min(int(self.spk_end_layer), total_decoder_blocks))
            for idx in range(total_decoder_blocks):
                self.spk_block_enabled[idx] = (
                    self.spk_start_layer <= idx < self.spk_end_layer
                    and ((idx - self.spk_start_layer) % self.spk_every == 0)
                )
                self.spk_concat_block_enabled[idx] = bool(self.spk_cond_use_concat and self.spk_block_enabled[idx])
                self.spk_film_block_enabled[idx] = bool(self.use_stage_speaker_film and self.spk_block_enabled[idx])
        else:
            self.spk_start_layer = 0
            self.spk_end_layer = 0

        if self.f0_condition:
            self.f0_width_list = list(reversed(f0_width_list[0]))
            # if not self.zero_out_all_unvoiced:
            #     print(colored(f"Introducing null f0 embeddings for unvoiced frames", "yellow", attrs=['bold']))
            #     self.f0_null_embeddings = nn.ParameterList()
            #     for w in self.f0_width_list:
            #         self.f0_null_embeddings.append(nn.Parameter(torch.randn(1, w, 1)))
            self.f0_start_layer = max(0, min(int(self.f0_start_layer), total_decoder_blocks))
            if self.f0_end_layer is None:
                self.f0_end_layer = total_decoder_blocks
            else:
                self.f0_end_layer = max(self.f0_start_layer, min(int(self.f0_end_layer), total_decoder_blocks))
            for idx in range(total_decoder_blocks):
                self.f0_block_enabled[idx] = (
                    self.f0_start_layer <= idx < self.f0_end_layer
                    and ((idx - self.f0_start_layer) % self.f0_every == 0)
                )
        else:
            self.f0_start_layer = 0
            self.f0_end_layer = 0

        print(colored(
            "CodecDecoder init: "
            f"decoder_type={self.decoder_type}, in_channels={in_channels}, "
            f"upsample_initial_channel={upsample_initial_channel}, up_ratios={list(up_ratios)}, "
            f"hop_length={self.hop_length}, quantizer_type={quantizer_type}, "
            f"speaker_condition={speaker_condition}, f0_condition={self.f0_condition}, "
            f"spk_layer_range=({self.spk_start_layer}, {self.spk_end_layer}), spk_every={self.spk_every}, "
            f"f0_layer_range=({self.f0_start_layer}, {self.f0_end_layer}), f0_every={self.f0_every}, "
            f"f0_speaker_condition={f0_speaker_condition}, "
            f"use_stage_speaker_film={use_stage_speaker_film}, use_mhca={use_mhca}, "
            f"activation_type={activation_type}",
            "cyan",
            attrs=["bold"],
        ))
        if self.f0_width_list:
            print(colored(f"CodecDecoder f0_width_list(reversed)={self.f0_width_list}", "cyan"))
            print(colored(f"CodecDecoder f0 enabled on decoder blocks {[idx for idx, enabled in enumerate(self.f0_block_enabled) if enabled]}", "cyan"))
        if speaker_condition:
            print(colored(f"CodecDecoder speaker enabled on decoder blocks {[idx for idx, enabled in enumerate(self.spk_block_enabled) if enabled]}", "cyan"))
        if any(self.spk_concat_block_enabled):
            print(colored(f"CodecDecoder speaker concat enabled on decoder blocks {[idx for idx, enabled in enumerate(self.spk_concat_block_enabled) if enabled]}", "cyan"))
        if any(self.spk_film_block_enabled):
            print(colored(f"CodecDecoder speaker FiLM enabled on decoder blocks {[idx for idx, enabled in enumerate(self.spk_film_block_enabled) if enabled]}", "cyan"))

        if self.decoder_type != 'default':
            raise ValueError(
                "This open SDPCodec config surface supports only "
                "codec_decoder.decoder_type='default'."
            )

        channels = upsample_initial_channel
        layers = [WNConv1d(in_channels, channels, kernel_size=7, padding=3)]
        
        if self.temporal.use:
            tt = self.temporal.backbone
            if tt == 'lstm':
                print(colored(
                    f"Using LSTM with {self.temporal.effective_num_layers()} layers and bidirectional={self.temporal.bidirectional}",
                    "blue",
                    attrs=['bold'],
                ))
                layers += [self.temporal.build_res_path(channels)]
            else:
                raise ValueError(
                    f"Unsupported decoder temporal type: {tt}. "
                    f"Supported: 'lstm'."
                )
        
        for i, stride in enumerate(up_ratios):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            # Per-block speaker conditioning (MHCA speaker path), and
            # speaker-modulated (joint speaker+F0 time-varying) F0 conditioning
            # wherever F0 conditioning is active.
            block_speaker_condition = bool(speaker_condition and self.spk_block_enabled[i])
            block_f0_condition = bool(self.f0_condition and self.f0_block_enabled[i])
            block_f0_speaker_condition = bool(f0_speaker_condition and block_f0_condition)
            layers += [DecoderBlock(input_dim, output_dim, stride, dilations, block_speaker_condition, condition_dim=condition_dim, f0_condition=block_f0_condition,
                                    f0_condition_dim=self.f0_width_list[i + (len(self.f0_width_list) - len(self.up_ratios))] if block_f0_condition else None,
                                    f0_speaker_condition=block_f0_speaker_condition,
                                    activation_type=activation_type,
                                    leaky_relu_params=leaky_relu_params,
                                    use_split_condition_optimization=self.use_split_condition_optimization,
                                    )]

        final_speaker_condition = bool(speaker_condition and self.spk_block_enabled and self.spk_block_enabled[-1])
        activation = build_codec_activation(
            dim=output_dim,
            activation_type=activation_type,
            leaky_relu_params=leaky_relu_params,
            speaker_condition=final_speaker_condition,
            condition_dim=condition_dim,
            alpha_logscale=snake_logscale,
        )
        # elif speaker_condition and f0_speaker_condition:
        #     activation = Activation1dWithCondition(activation=activations.SnakeBetaWithTimeVaryingCondition(output_dim, condition_dim + (self.f0_width_list[-1] if f0_condition else 0), alpha_logscale=True))
        layers += [
            activation,
            WNConv1d(output_dim, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

        self.speaker_stage_fusers = nn.ModuleList()
        self.speaker_stage_film = nn.ModuleList()
        for i, _stride in enumerate(up_ratios):
            input_dim = channels // 2**i
            # Keep legacy default-decoder behavior: speaker concat happens only
            # inside DecoderBlock, not through an additional pre-block fuser.
            self.speaker_stage_fusers.append(nn.Identity())

            if self.spk_film_block_enabled[i]:
                self.speaker_stage_film.append(
                    nn.Linear(condition_dim, input_dim * 2)
                )
            else:
                self.speaker_stage_film.append(nn.Identity())
        
        # Cross-Attention Block for each DecoderBlock
        self.mhca_block_enabled = [False] * len(up_ratios)
        if use_mhca:
            print(colored("Using Cross-Attention Blocks for speaker conditioning at each DecoderBlock", "cyan", attrs=['bold']))
            self.mhca_list = nn.ModuleList()
            mhca_cfg = CodecDecoderSpeakerConditionConfig(
                use=use_mhca,
                type="mhca",
                num_heads=mhca_num_heads,
                dropout=mhca_dropout,
                start_layer=mhca_start_layer,
                end_layer=mhca_end_layer,
                every=mhca_every,
            )
            start, end, every = mhca_cfg.resolve_block_range(len(up_ratios))
            for i, stride in enumerate(up_ratios):
                input_dim = channels // 2**i
                enabled = start <= i < end and ((i - start) % every == 0)
                self.mhca_block_enabled[i] = enabled
                self.mhca_list.append(
                    CrossAttentionBlock(
                        query_dim=input_dim,
                        key_dim=mhca_key_dim,  # x_quantized_projected dim (out_dim or latent_dim)
                        num_heads=mhca_num_heads,
                        ffn_hidden_dim=None,  # defaults to 4 * query_dim
                        dropout=mhca_dropout,
                        use_sdpa=mhca_use_sdpa,
                    )
                    if enabled else nn.Identity()
                )
            print(colored(f"Created {sum(self.mhca_block_enabled)} Cross-Attention Blocks for DecoderBlocks (key_dim={mhca_key_dim})", "cyan"))
            print(colored(f"MHCA enabled on decoder blocks {[idx for idx, enabled in enumerate(self.mhca_block_enabled) if enabled]}", "cyan"))
        else:
            self.mhca_list = None
        
        self.reset_parameters()
        self.latest_decoder_aux = None

    def forward(self, x, total_step=None, vq=True, spk_cond=None, f0_conds=None, vuv=None):
        if vq is True:
            x, q, commit_loss, perplexity, active_num = self.quantizer(x, total_step=total_step, produce_targets=True)
            return x, q, commit_loss, perplexity, active_num

        # Parse speaker conditioning: when available, tuple is (global_spk, speaker_tokens)
        if isinstance(spk_cond, tuple):
            gq_vector, x_quantized = spk_cond
            spk_cond_global = gq_vector
        else:
            spk_cond_global = spk_cond
            x_quantized = None

        # x = self.model(x, condition=condition)
        # decoder_block_num = 0
        decoder_block_num = len(self.f0_width_list) - len(self.up_ratios) if self.f0_condition else 0
        decoder_block_idx = 0
        for i, layer in enumerate(self.model):
            if isinstance(layer, Activation1dWithCondition):
                x = layer(x, spk_cond_global)
            elif isinstance(layer, DecoderBlock):
                if (
                    decoder_block_idx < len(self.spk_film_block_enabled)
                    and self.spk_film_block_enabled[decoder_block_idx]
                    and spk_cond_global is not None
                ):
                    gamma_beta = self.speaker_stage_film[decoder_block_idx](spk_cond_global)
                    gamma, beta = gamma_beta.chunk(2, dim=1)
                    x = x * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

                # Apply Cross-Attention Block before each DecoderBlock
                if (
                    self.use_mhca
                    and x_quantized is not None
                    and self.mhca_list is not None
                    and decoder_block_idx < len(self.mhca_block_enabled)
                    and self.mhca_block_enabled[decoder_block_idx]
                ):
                    x = self.mhca_list[decoder_block_idx](x, x_quantized)
                
                # f0_cond = f0_conds[decoder_block_num].detach() if self.f0_condition else None
                f0_cond = (
                    f0_conds[decoder_block_num]
                    if self.f0_condition and self.f0_block_enabled[decoder_block_idx]
                    else None
                )
                x = layer(x, spk_cond_global, f0_cond=f0_cond)
                decoder_block_num += 1
                decoder_block_idx += 1
            else:
                x = layer(x)
        
        self.latest_decoder_aux = None
        return x

    def vq2emb(self, vq):
        self.quantizer = self.quantizer.eval()
        x = self.quantizer.vq2emb(vq)
        return x

    def get_emb(self):
        self.quantizer = self.quantizer.eval()
        embs = self.quantizer.get_emb()
        return embs

    def inference_vq(self, vq):
        x = vq[None,:,:]
        x = self.model(x)
        return x

    def inference_0(self, x):
        x, q, loss, perp = self.quantizer(x)
        x = self.model(x)
        return x, None
    
    def inference(self, x):
        x = self.model(x)
        return x, None


    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)

    def get_latest_decoder_aux(self):
        return self.latest_decoder_aux
