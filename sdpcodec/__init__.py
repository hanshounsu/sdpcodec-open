"""Importable SDPCodec package classes."""

__all__ = [
    "SdpCodecDataModule",
    "SdpCodecLightningModule",
    "SdpCodecSystem",
]


def __getattr__(name):
    if name == "SdpCodecDataModule":
        from sdpcodec.data import SdpCodecDataModule

        return SdpCodecDataModule
    if name in {"SdpCodecLightningModule", "SdpCodecSystem"}:
        from sdpcodec.system import SdpCodecLightningModule, SdpCodecSystem

        return {
            "SdpCodecLightningModule": SdpCodecLightningModule,
            "SdpCodecSystem": SdpCodecSystem,
        }[name]
    raise AttributeError(f"module 'sdpcodec' has no attribute {name!r}")
