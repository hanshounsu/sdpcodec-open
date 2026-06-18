"""Lightning module wrapper for SDPCodec."""

from __future__ import annotations

from ptl.sdpcodec.lightning_module import SdpCodecLightningModule as _CoreSdpCodecLightningModule


class SdpCodecLightningModule(_CoreSdpCodecLightningModule):
    """Joint content/F0 codec system used by SDPCodec."""


SdpCodecSystem = SdpCodecLightningModule
