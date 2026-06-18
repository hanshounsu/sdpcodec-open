# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.amp import autocast

from typing import Any, Dict, List, Tuple, Optional
from vq.speaker.perceiver_encoder import (
    MemoryCrossAttentionEncoder,
    PerceiverResampler,
    SpeakerTokenMixer,
)

from termcolor import colored

from vq.speaker.wavlm.WavLM import WavLM, WavLMConfig


"""
x-vector + d-vector
"""


class ConvPromptPrenet(nn.Module):
    """
    Lightweight adaptation of vec2wav2.0 ConvPromptPrenet to compress WavLM prompts.

    Args:
        in_channels (int): input feature dimension.
        out_channels (int): output feature dimension.
        conv_layers (List[Tuple[int, int, int, int]]): sequence of (dim, kernel, stride, padding).
        dropout (float): dropout probability applied after each conv.
        skip_connections (bool): whether to use residual connections.
        residual_scale (float): scaling factor for residual paths.
        activation (Callable): activation factory (default nn.ReLU).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_layers: Optional[List[Tuple[int, int, int, int]]] = None,
        dropout: float = 0.1,
        skip_connections: bool = True,
        residual_scale: float = 0.25,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        if conv_layers is None or len(conv_layers) == 0:
            hidden_dim = max(out_channels, in_channels // 2)
            conv_layers = [
                (hidden_dim, 3, 1, 1),
                (out_channels, 3, 1, 1),
            ]

        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale) if skip_connections else 1.0
        act_factory = activation if activation is not None else nn.ReLU

        layers: List[nn.Module] = []
        residual_proj: List[Optional[nn.Module]] = []
        in_dim = int(in_channels)

        for dim, kernel, stride, padding in conv_layers:
            dim = int(dim)
            block = nn.Sequential(
                nn.Conv1d(in_dim, dim, kernel_size=kernel, stride=stride, padding=padding, bias=True),
                nn.Dropout(p=dropout),
                nn.GroupNorm(1, dim),
                act_factory(),
            )
            layers.append(block)

            if skip_connections and dim != in_dim:
                residual_proj.append(nn.Conv1d(in_dim, dim, kernel_size=1, bias=False))
            else:
                residual_proj.append(None)

            in_dim = dim

        if in_dim != out_channels:
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_dim, out_channels, kernel_size=1, bias=True),
                    nn.Dropout(p=dropout),
                    nn.GroupNorm(1, out_channels),
                    act_factory(),
                )
            )
            residual_proj.append(nn.Conv1d(in_dim, out_channels, kernel_size=1, bias=False) if skip_connections else None)

        self.conv_layers = nn.ModuleList(layers)
        self.residual_proj = nn.ModuleList(residual_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for proj, conv in zip(self.residual_proj, self.conv_layers):
            residual = x
            x = conv(x)
            if self.skip_connections:
                if proj is not None:
                    residual = proj(residual)
                x = (x + residual) * self.residual_scale
        return x


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _stack_cfg_enabled(stack_cfg: Any) -> bool:
    return bool(stack_cfg is not None and _cfg_get(stack_cfg, 'use', False))


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {'', 'null', 'none'}:
            return None
        return int(stripped)
    return int(value)


def _normalize_stack_stage_name(raw_stage: Any) -> str:
    if raw_stage is None:
        raise ValueError("speaker_encoder.stack stage cannot be None")

    text = str(raw_stage).strip().lower().replace('-', '_')
    if any(char.isdigit() for char in text):
        raise ValueError(
            f"speaker_encoder.stack stage '{raw_stage}' should not include token counts. "
            "Use stack.layers=[mca,pe,tm,pe] and stack.token_nums=[64,64,null,16]."
        )

    raw_name = text.replace('_', '')
    stage_name = {
        'mca': 'mca',
        'memory': 'mca',
        'memorycattn': 'mca',
        'memorycrossattention': 'mca',
        'memorycrossattn': 'mca',
        'pe': 'pe',
        'perceiver': 'pe',
        'perceiverresampler': 'pe',
        'resampler': 'pe',
        'tm': 'tm',
        'mixer': 'tm',
        'tokenmixer': 'tm',
        'selfattn': 'tm',
        'selfattention': 'tm',
    }.get(raw_name)
    if stage_name is None:
        raise ValueError(
            f"Unsupported speaker_encoder.stack stage '{raw_stage}'. "
            "Expected one of mca / pe / tm."
        )

    return stage_name


def _parse_speaker_stack(stack_cfg: Any) -> List[Dict[str, Any]]:
    if not _stack_cfg_enabled(stack_cfg):
        return []

    raw_layers = (
        _cfg_get(stack_cfg, 'layers')
        or _cfg_get(stack_cfg, 'types')
        or _cfg_get(stack_cfg, 'stages')
        or _cfg_get(stack_cfg, 'modules')
        or _cfg_get(stack_cfg, 'specs')
    )
    if raw_layers is None or len(raw_layers) == 0:
        raise ValueError(
            "speaker_encoder.stack.use=True requires non-empty stack.layers "
            "(or stack.types / stack.stages / stack.modules / stack.specs)."
        )

    raw_token_nums = _cfg_get(stack_cfg, 'token_nums', _cfg_get(stack_cfg, 'tokens'))
    raw_depths = _cfg_get(stack_cfg, 'depths')
    raw_latent_dims = _cfg_get(stack_cfg, 'latent_dims', _cfg_get(stack_cfg, 'dims'))
    if raw_token_nums is None:
        raise ValueError(
            "speaker_encoder.stack.use=True requires stack.token_nums with the same length "
            "as stack.layers. Use null for tm stages."
        )
    if len(raw_token_nums) != len(raw_layers):
        raise ValueError(
            "speaker_encoder.stack.token_nums must have the same length as stack.layers"
        )
    if raw_depths is not None and len(raw_depths) != len(raw_layers):
        raise ValueError(
            "speaker_encoder.stack.depths must have the same length as stack.layers"
        )
    if raw_latent_dims is not None and len(raw_latent_dims) != len(raw_layers):
        raise ValueError(
            "speaker_encoder.stack.latent_dims must have the same length as stack.layers"
        )

    default_depth = int(_cfg_get(stack_cfg, 'depth', 2))
    num_heads = int(_cfg_get(stack_cfg, 'num_heads', 8))
    dim_head = int(_cfg_get(stack_cfg, 'dim_head', 64))
    ff_mult = int(_cfg_get(stack_cfg, 'ff_mult', 4))
    dropout = float(_cfg_get(stack_cfg, 'dropout', 0.0))
    use_flash_attn = bool(_cfg_get(stack_cfg, 'use_flash_attn', False))
    tm_type = str(_cfg_get(stack_cfg, 'tm_type', 'self_attn')).strip().lower()
    if tm_type not in {'self_attn', 'selfattention', 'attn'}:
        raise ValueError(
            f"Unsupported speaker_encoder.stack.tm_type: {tm_type}. "
            "Only 'self_attn' is supported for now."
        )

    stages: List[Dict[str, Any]] = []
    for idx, raw_stage in enumerate(raw_layers):
        stage_type = _normalize_stack_stage_name(raw_stage)
        token_num = _coerce_optional_int(raw_token_nums[idx])

        if stage_type in {'mca', 'pe'}:
            if token_num is None:
                raise ValueError(
                    f"speaker_encoder.stack stage '{raw_stage}' requires a numeric token_num."
                )
            token_num = int(token_num)
        elif token_num is not None:
            raise ValueError("speaker_encoder.stack tm stage must use null token_num")

        depth = default_depth if raw_depths is None else int(raw_depths[idx])
        if depth < 1:
            raise ValueError("speaker_encoder.stack stage depth must be >= 1")
        latent_dim = None if raw_latent_dims is None else _coerce_optional_int(raw_latent_dims[idx])
        if latent_dim is not None and latent_dim < 1:
            raise ValueError("speaker_encoder.stack stage latent_dim must be >= 1")

        stages.append(
            {
                'type': stage_type,
                'token_num': token_num,
                'depth': depth,
                'latent_dim': latent_dim,
                'num_heads': num_heads,
                'dim_head': dim_head,
                'ff_mult': ff_mult,
                'dropout': dropout,
                'use_flash_attn': use_flash_attn,
            }
        )

    return stages


def _infer_final_token_num_from_stack(stages: List[Dict[str, Any]]) -> Optional[int]:
    fixed_token_num: Optional[int] = None
    for stage in stages:
        if stage['type'] == 'pe':
            fixed_token_num = int(stage['token_num'])
    return fixed_token_num


def _infer_final_latent_dim_from_stack(stages: List[Dict[str, Any]]) -> Optional[int]:
    final_latent_dim: Optional[int] = None
    for stage in stages:
        stage_latent_dim = stage.get('latent_dim')
        if stage_latent_dim is not None:
            final_latent_dim = int(stage_latent_dim)
    return final_latent_dim


def resolve_speaker_encoder_token_dim(spkcfg: Any) -> int:
    stack_cfg = _cfg_get(spkcfg, 'stack')
    base_latent_dim = int(_cfg_get(spkcfg, 'latent_dim'))
    stages = _parse_speaker_stack(stack_cfg)
    if len(stages) == 0:
        return base_latent_dim

    final_latent_dim = _infer_final_latent_dim_from_stack(stages)
    if final_latent_dim is None:
        return base_latent_dim
    return int(final_latent_dim)


class _ZeroPosConv(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class SpeakerEncoder(nn.Module):
    """

    Args:
        input_dim (int): acoustic feature dimension
        out_dim (int): output dimension of x-vector and d-vector
        latent_dim (int): latent dimension before speaker projection
        token_num (int): sequence length of speaker tokens

    Return:
        speaker_embs: (B, T2, out_dim)
    """

    def __init__(
        self,
        mel_params: Optional[Any] = None,
        speaker_encoder_type: str = 'wavlm',
        use_perceiver_encoder: bool = True,
        use_memory_cattn: bool = False,
        input_dim: int = 100,
        out_dim: int = 512,
        latent_dim: int = 128,
        token_num: int = 32,
        discretize_memory_attn: bool = False,
        memory_attn_codebook_size: int = 128,
        memory_attn_share_across_heads: bool = True,
        use_normalized_f0: bool = False,
        use_quantizer: bool = True,
        stack: Optional[Any] = None,
        # WavLM specific (optional)
        wavlm_checkpoint: str = None,
        wavlm_output_layer: int = 6,
        freeze_wavlm: bool = True,
        perceiver_use_flash_attn: bool = False,
        frozen_wavlm_inference_mode: bool = False,
        frozen_wavlm_force_fp32: bool = False,
        wavlm_disable_pos_conv: bool = False,
        wavlm_disable_relative_position_bias: bool = False,
    ):
        super(SpeakerEncoder, self).__init__()
        self.speaker_encoder_type = speaker_encoder_type
        self.token_stack = nn.ModuleList()
        self.stack_stages = _parse_speaker_stack(stack)
        self.use_stack = len(self.stack_stages) > 0
        self.use_perceiver_encoder = bool(use_perceiver_encoder or self.use_stack)
        self.use_memory_cattn = use_memory_cattn
        self.perceiver_use_flash_attn = bool(perceiver_use_flash_attn)
        self.use_normalized_f0 = use_normalized_f0
        self.use_quantizer = use_quantizer
        self.out_dim = out_dim
        self.base_latent_dim = int(latent_dim)
        self.final_latent_dim = int(latent_dim)

        print(colored(f'Importing {speaker_encoder_type} speaker encoder', 'red', attrs=['bold']))
        dim_context = None

        if speaker_encoder_type == 'wavlm':
            if WavLM is None or WavLMConfig is None:
                raise ImportError("Failed to import bundled WavLM implementation.")
            # Resolve checkpoint
            ckpt_path = wavlm_checkpoint or "pretrained/WavLM-Large.pt"
            print(colored(f"Loading WavLM from {ckpt_path}", "yellow"))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self.wavlm_cfg = WavLMConfig(ckpt['cfg'])
            self.wavlm_model = WavLM(self.wavlm_cfg)
            self.wavlm_model.load_state_dict(ckpt['model'])
            if wavlm_disable_pos_conv:
                self.wavlm_model.encoder.pos_conv = _ZeroPosConv()
                print(colored("WavLM absolute positional conv disabled", "yellow"))
            if wavlm_disable_relative_position_bias:
                self.wavlm_model.encoder.relative_position_embedding = False
                for layer in self.wavlm_model.encoder.layers:
                    attn = getattr(layer, 'self_attn', None)
                    if attn is None:
                        continue
                    if hasattr(attn, 'has_relative_attention_bias'):
                        attn.has_relative_attention_bias = False
                    if hasattr(attn, 'gru_rel_pos'):
                        attn.gru_rel_pos = False
                print(colored("WavLM relative positional attention bias disabled", "yellow"))
            # freezing policy
            if freeze_wavlm:
                for p in self.wavlm_model.parameters():
                    p.requires_grad = False
                self.wavlm_model.eval()
            self.freeze_wavlm = bool(freeze_wavlm)
            self.frozen_wavlm_inference_mode = bool(frozen_wavlm_inference_mode)
            self.frozen_wavlm_force_fp32 = bool(frozen_wavlm_force_fp32)
            self.use_inference_mode_for_frozen_wavlm = (
                self.freeze_wavlm
                and (
                    self.frozen_wavlm_inference_mode
                    or os.environ.get("SPEAKER_WAVLM_INFERENCE_MODE", "0") == "1"
                )
            )
            self.wavlm_output_layer = int(wavlm_output_layer)
            # Use normalize flag from checkpoint config (not external param)
            self.wavlm_normalize = bool(self.wavlm_cfg.normalize)
            self.wavlm_feature_dim = int(self.wavlm_cfg.encoder_embed_dim)
            print(colored(f"WavLM normalize from checkpoint cfg: {self.wavlm_normalize}", "yellow"))
            if self.use_inference_mode_for_frozen_wavlm:
                print(colored("Frozen WavLM runs under torch.inference_mode()", "yellow"))
            if self.freeze_wavlm and self.frozen_wavlm_force_fp32:
                print(colored("Frozen WavLM runs under fp32 autocast-disabled mode", "yellow"))
            
            # Apply torch.compile for faster inference (PyTorch 2.0+)
            # self.wavlm_model = torch.compile(self.wavlm_model, mode='reduce-overhead')
            # print(colored("✓ torch.compile applied to WavLM model", "green"))
            
            # x-vector projection head (global pooled WavLM -> out_dim)
            self.x_project = nn.Linear(self.wavlm_feature_dim, out_dim)
            dim_context = self.wavlm_feature_dim
        else:
            raise ValueError(f"SdpCodec supports only speaker_encoder_type='wavlm', got {speaker_encoder_type!r}")
        
        self.perceiver_sampler = None
        self.memory_encoder = None
        self.quantizer = None
        self.project = None
        self.prompt_prenet = None
        self.final_token_num = None

        if use_perceiver_encoder or self.use_stack:
            print(colored("Using Perceiver encoder in Speaker Encoder", "yellow", attrs=['bold']))
            if self.use_stack:
                stage_labels = []
                current_dim = dim_context
                for stage in self.stack_stages:
                    stage_type = stage['type']
                    stage_token_num = stage['token_num']
                    stage_latent_dim = int(stage['latent_dim'] if stage['latent_dim'] is not None else self.base_latent_dim)
                    common_kwargs = dict(
                        dim=stage_latent_dim,
                        depth=stage['depth'],
                        dim_head=stage['dim_head'],
                        heads=stage['num_heads'],
                        ff_mult=stage['ff_mult'],
                        use_flash_attn=stage['use_flash_attn'],
                    )

                    if stage_type == 'mca':
                        module = MemoryCrossAttentionEncoder(
                            dim_context=current_dim,
                            num_latents=stage_token_num,
                            discretize_attn=discretize_memory_attn,
                            attn_codebook_size=memory_attn_codebook_size,
                            attn_share_across_heads=memory_attn_share_across_heads,
                            **common_kwargs,
                        )
                        stage_labels.append(f"mca(kv={stage_token_num},d={stage_latent_dim})")
                        current_dim = stage_latent_dim
                    elif stage_type == 'pe':
                        module = PerceiverResampler(
                            dim_context=current_dim,
                            num_latents=stage_token_num,
                            **common_kwargs,
                        )
                        stage_labels.append(f"pe(n={stage_token_num},d={stage_latent_dim})")
                        current_dim = stage_latent_dim
                    elif stage_type == 'tm':
                        mixer = SpeakerTokenMixer(
                            dim=stage_latent_dim,
                            depth=stage['depth'],
                            dim_head=stage['dim_head'],
                            heads=stage['num_heads'],
                            ff_mult=stage['ff_mult'],
                            dropout=stage['dropout'],
                            use_flash_attn=stage['use_flash_attn'],
                        )
                        module = nn.Sequential(nn.Linear(current_dim, stage_latent_dim), mixer) if current_dim != stage_latent_dim else mixer
                        stage_labels.append(f"tm(d={stage_latent_dim})")
                        current_dim = stage_latent_dim
                    else:
                        raise ValueError(f"Unsupported speaker stack stage type: {stage_type}")

                    self.token_stack.append(module)

                self.final_token_num = _infer_final_token_num_from_stack(self.stack_stages)
                self.final_latent_dim = int(current_dim)
                if self.final_token_num is None:
                    raise ValueError(
                        "speaker_encoder.stack must contain at least one PE stage so the "
                        "final speaker token count is fixed for projection."
                    )
                print(
                    colored(
                        f"Speaker encoder stack enabled: {' -> '.join(stage_labels)} "
                        f"(final_token_num={self.final_token_num})",
                        "yellow",
                        attrs=['bold'],
                    )
                )
            else:
                if self.use_memory_cattn:
                    print(colored("Speaker encoder memory cross-attention is enabled", "yellow", attrs=['bold']))
                    self.memory_encoder = MemoryCrossAttentionEncoder(
                        dim=self.base_latent_dim,
                        dim_context=dim_context,
                        num_latents=token_num,
                        use_flash_attn=self.perceiver_use_flash_attn,
                        discretize_attn=discretize_memory_attn,
                        attn_codebook_size=memory_attn_codebook_size,
                        attn_share_across_heads=memory_attn_share_across_heads,
                    )
                self.perceiver_sampler = PerceiverResampler(
                    dim=self.base_latent_dim,
                    dim_context=self.base_latent_dim if self.use_memory_cattn else dim_context,
                    num_latents=token_num,
                    use_flash_attn=self.perceiver_use_flash_attn,
                )
                self.final_token_num = int(token_num)
                self.final_latent_dim = self.base_latent_dim

            if self.use_quantizer:
                raise ValueError(
                    "The open SDPCodec baseline uses speaker_encoder.use_quantizer=false. "
                    "Speaker quantizers are not part of the supported config surface."
                )
            
            self.project = nn.Linear(self.final_latent_dim * int(self.final_token_num), out_dim)
        else:
            if dim_context is None:
                raise ValueError("dim_context must be defined when use_perceiver_encoder=False")
            print(colored("Perceiver disabled: using Conv1d prompt prenet instead of quantization", "yellow", attrs=['bold']))
            mid_dim = max(self.base_latent_dim, dim_context // 2)
            conv_cfg = [
                (dim_context, 3, 1, 1),
                (mid_dim, 5, 1, 2),
                (self.base_latent_dim, 3, 1, 1),
            ]
            self.prompt_prenet = ConvPromptPrenet(
                in_channels=dim_context,
                out_channels=self.base_latent_dim,
                conv_layers=conv_cfg,
                dropout=0.1,
                skip_connections=True,
                residual_scale=0.25,
                activation=nn.ReLU,
            )
            self.final_latent_dim = self.base_latent_dim

    def init_mel_transformer(self, cfg):
        """
        Initializes the MelSpectrogram transformer based on the provided configuration.

        Args:
            config (dict): Configuration parameters for MelSpectrogram.
        """
        import torchaudio.transforms as TT
        self.mel_transformer = TT.MelSpectrogram(
            cfg.sample_rate,
            cfg.n_fft,
            cfg.win_length,
            cfg.hop_length,
            cfg.mel_fmin,
            cfg.mel_fmax,
            n_mels=cfg.num_mels,
            power=1,
            norm="slaney",
            mel_scale="slaney",
        )

    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Speaker token quantization is not supported by this SDPCodec baseline.")

    def get_indices(self, _mels: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Speaker token quantization is not supported by this SDPCodec baseline.")

    def _encode_speaker_tokens(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T_ctx, D_ctx) backbone speaker features

        Returns:
            Speaker tokens with fixed token length after optional memory cross-attn
            and Perceiver resampling: (B, token_num, latent_dim)
        """
        x = features
        if self.use_stack:
            if len(self.token_stack) == 0:
                raise RuntimeError("speaker_encoder.stack is enabled but no token stack modules were built.")
            for layer in self.token_stack:
                x = layer(x)
            return x
        if self.use_memory_cattn:
            assert self.memory_encoder is not None, "memory_encoder must be defined when use_memory_cattn=True."
            x = self.memory_encoder(x)  # (B, T_ctx, latent_dim)
        assert self.perceiver_sampler is not None, "perceiver_sampler must be defined when use_perceiver_encoder=True."
        return self.perceiver_sampler(x)  # (B, token_num, latent_dim)

    def forward(self, ref_wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            ref_wav: (B, T1)

        Return:
            x_vector: (B, out_dim) continuous speaker embedding from the backbone
            d_vector: (B, out_dim) projected embedding after Perceiver / memory cross-attn (+ optional VQ)
            x_quantized: (B, latent_dim, T_spk) speaker token sequence before global pooling
        """
        # mels = mels.transpose(1,2)

        wav = ref_wav  # [B, T]
        if self.wavlm_normalize:
            wav = torch.nn.functional.layer_norm(wav, wav.shape)
        use_fp32_island = (
            wav.is_cuda
            and self.freeze_wavlm
            and self.frozen_wavlm_force_fp32
        )
        autocast_ctx = (
            autocast(device_type=wav.device.type, enabled=False)
            if use_fp32_island
            else nullcontext()
        )
        if self.use_inference_mode_for_frozen_wavlm:
            with autocast_ctx:
                with torch.inference_mode():
                    features = self.wavlm_model.extract_features(
                        wav,
                        output_layer=self.wavlm_output_layer,
                    )[0]
            # Inference tensors cannot be saved for backward by downstream trainable layers.
            features = features.clone()
        else:
            with autocast_ctx:
                with torch.no_grad():
                    features = self.wavlm_model.extract_features(
                        wav,
                        output_layer=self.wavlm_output_layer,
                    )[0]
        x_vector = self.x_project(features.mean(dim=1))
        
        if self.use_perceiver_encoder:
            # Speaker token flow:
            # - default: backbone features -> PerceiverResampler -> fixed token_num tokens
            # - use_memory_cattn=True: backbone features -> MemoryCrossAttentionEncoder
            #   -> PerceiverResampler -> fixed token_num tokens
            x = self._encode_speaker_tokens(features).transpose(1, 2)  # (B, latent_dim, token_num)

            z_q = x  # (B, latent_dim, token_num)
            
            # x_vector and d_vector have the same shape (B, out_dim), but come from
            # different stages. x_vector is the backbone global embedding, while
            # d_vector is rebuilt from the fixed-length speaker token sequence.
            pooled = z_q.reshape(z_q.shape[0], -1)  # (B, latent_dim * token_num)
            gq_vector = self.project(pooled)  # (B, out_dim)
            x_quantized_projected = z_q  # (B, latent_dim, token_num)
        else:
            assert self.prompt_prenet is not None, "prompt_prenet must be defined when use_perceiver_encoder=False."
            feats = features.transpose(1, 2)  # (B, D, T)
            x_quantized_projected = self.prompt_prenet(feats)  # (B, latent_dim, T_spk)
            return x_vector, None, x_quantized_projected, None

        return x_vector, gq_vector, x_quantized_projected, None
    
    def tokenize(self, _mels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("SdpCodec uses WavLM speaker prompts. Use tokenize_wav().")

    def tokenize_wav(self, wav: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Speaker token quantization is not supported by this SDPCodec baseline.")
    
    def detokenize(self, indices: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Speaker token quantization is not supported by this SDPCodec baseline.")
