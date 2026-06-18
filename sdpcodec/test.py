"""Run SDPCodec dataset-level inference/evaluation from a checkpoint."""

from __future__ import annotations

from typing import Any

import hydra
import pytorch_lightning as pl
from omegaconf import open_dict

from sdpcodec.data import SdpCodecDataModule
from sdpcodec.system import SdpCodecLightningModule


@hydra.main(version_base=None, config_path="../configs", config_name="sdpcodec_vqw2v_rvq300")
def test(cfg: Any) -> None:
    if not getattr(cfg, "ckpt", None):
        raise ValueError("Set ckpt=/path/to/checkpoint.ckpt for dataset-level inference.")

    with open_dict(cfg):
        cfg.train.wandb_enabled = False
        cfg.train.use_val_utmos = False
        cfg.train.use_val_wer = False

    datamodule = SdpCodecDataModule(cfg)
    model = SdpCodecLightningModule(cfg)
    trainer = pl.Trainer(
        accelerator=getattr(cfg.train.trainer, "accelerator", "auto"),
        devices=getattr(cfg.train.trainer, "devices", 1),
        precision=getattr(cfg.train.trainer, "precision", "32-true"),
        logger=False,
        enable_checkpointing=False,
    )
    trainer.test(model, datamodule=datamodule, ckpt_path=str(cfg.ckpt))


if __name__ == "__main__":
    test()
