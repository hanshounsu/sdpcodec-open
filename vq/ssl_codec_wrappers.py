from contextlib import nullcontext
from typing import Optional

import torch.nn as nn
from torch.amp import autocast

from .module import WNConv1d, EncoderBlock, build_codec_activation
from .temporal_config import CodecTemporalConfig


class VQW2VCodecEncoderWrapper(nn.Module):
    """VQ-Wav2Vec feature wrapper used by the paper reproduction config."""

    def __init__(
        self,
        encoder,
        activation_type="SnakeBeta",
        leaky_relu_params=None,
        temporal: Optional[CodecTemporalConfig] = None,
        up_ratios=(2,),
        dilations=(1, 3, 9),
        ngf=512,
        out_channels=512,
        encoder_force_fp32: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.temporal = temporal or CodecTemporalConfig()
        self.encoder_force_fp32 = bool(encoder_force_fp32)
        self.encoder_is_frozen = not any(
            param.requires_grad for param in self.encoder.parameters()
        )

        def make_activation(ch: int):
            return build_codec_activation(
                dim=ch,
                activation_type=activation_type,
                leaky_relu_params=leaky_relu_params,
                alpha_logscale=True,
                no_condition=True,
            )

        layers = []
        if self.temporal.use:
            layers.append(self.temporal.build(ngf))

        if up_ratios is not None:
            for stride in up_ratios:
                ngf *= 2
                layers.append(
                    EncoderBlock(
                        ngf,
                        stride=stride,
                        dilations=dilations,
                        activation_type=activation_type,
                        leaky_relu_params=leaky_relu_params,
                    )
                )

        layers += [
            make_activation(ngf),
            WNConv1d(ngf, out_channels, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x, spk_cond=None):
        use_fp32_island = (
            x.is_cuda
            and self.encoder_is_frozen
            and self.encoder_force_fp32
        )
        autocast_ctx = (
            autocast(device_type=x.device.type, enabled=False)
            if use_fp32_island
            else nullcontext()
        )
        if self.encoder_is_frozen:
            self.encoder.eval()
        with autocast_ctx:
            x = self.encoder(x)
        if isinstance(x, (tuple, list)):
            x = x[0]
        return self.block(x)


__all__ = [
    "VQW2VCodecEncoderWrapper",
]
