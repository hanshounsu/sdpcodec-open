import os
from contextlib import nullcontext

import torch
from torch import nn
from torch.amp import autocast

class VQW2VEncoder(nn.Module):
    """
    VQ-Wav2Vec encoder with two modes:
    - use_continuous=True (default): Output pre-quantization features (feature_extractor only).
    - use_continuous=False: Quantize to discrete indices, lookup pretrained codebook, output discrete features.
    Both modes output (B, C, T) with same C for compatibility with VQW2VCodecEncoderWrapper.
    """
    def __init__(
        self,
        feature_extractor,
        vector_quantizer,
        use_continuous=True,
        frozen_feature_extractor_force_fp32: bool = False,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.vector_quantizer = vector_quantizer
        self.use_continuous = use_continuous
        self.feature_extractor_frozen = not any(
            param.requires_grad for param in self.feature_extractor.parameters()
        )
        self.frozen_feature_extractor_force_fp32 = bool(
            frozen_feature_extractor_force_fp32
        )
        self.use_inference_mode_for_frozen_feature_extractor = (
            self.feature_extractor_frozen
            and os.environ.get("CODEC_VQW2V_INFERENCE_MODE", "0") == "1"
        )
        if self.use_inference_mode_for_frozen_feature_extractor:
            print("CODEC_VQW2V_INFERENCE_MODE=1: frozen VQ-Wav2Vec feature extractor runs under torch.inference_mode()")
        if (
            self.feature_extractor_frozen
            and self.frozen_feature_extractor_force_fp32
        ):
            print("Frozen VQ-Wav2Vec feature extractor runs under fp32 autocast-disabled mode")

    def forward(self, x):
        # x: (B, T) waveform
        use_fp32_island = (
            x.is_cuda
            and self.feature_extractor_frozen
            and self.frozen_feature_extractor_force_fp32
        )
        autocast_ctx = (
            autocast(device_type=x.device.type, enabled=False)
            if use_fp32_island
            else nullcontext()
        )
        if self.use_inference_mode_for_frozen_feature_extractor:
            with autocast_ctx:
                with torch.inference_mode():
                    features = self.feature_extractor(x)  # (B, C, T)
            # Inference tensors cannot be saved for backward by downstream trainable layers.
            features = features.clone()
        else:
            with autocast_ctx:
                features = self.feature_extractor(x)  # (B, C, T)
        if self.use_continuous:
            return features
        # Discrete: quantize to idx, lookup pretrained codebook, return discrete features
        with torch.no_grad():
            zq, _ = self.vector_quantizer.forward_idx(features)
        return zq
