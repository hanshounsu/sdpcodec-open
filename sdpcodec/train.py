"""Train the SDPCodec main reproduction config."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from sdpcodec.data import SdpCodecDataModule
from sdpcodec.system import SdpCodecLightningModule


class SamplerStateCheckpointCallback(pl.Callback):
    """Save DataModule sampler state so mid-epoch resumes stay aligned."""

    def on_save_checkpoint(self, trainer: pl.Trainer, _pl_module: pl.LightningModule, checkpoint: dict[str, Any]) -> None:
        datamodule = getattr(trainer, "datamodule", None)
        sampler = getattr(datamodule, "_train_sampler", None)
        if sampler is None:
            return
        try:
            checkpoint["datamodule_sampler_state"] = sampler.state_dict()
        except Exception:
            return


class RankZeroProgressBar(TQDMProgressBar):
    """Only show the progress bar on rank 0."""

    @property
    def _is_enabled(self) -> bool:
        return self.trainer.global_rank == 0


def _is_main_process() -> bool:
    for key in ("LOCAL_RANK", "RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value is not None:
            return int(value) == 0
    try:
        import torch.distributed as dist

        return not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        return True


def _device_count(devices: Any) -> int:
    if isinstance(devices, int):
        return devices
    if isinstance(devices, (list, tuple)):
        return len(devices)
    if isinstance(devices, str):
        if devices.strip().isdigit():
            return int(devices)
        if "," in devices:
            return len([item for item in devices.split(",") if item.strip()])
    return 1


def _configure_reproducibility(cfg: Any) -> None:
    seed_everything(int(getattr(cfg, "seed", 1024)), workers=True)

    deterministic = bool(getattr(cfg, "deterministic", False))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    train_cfg = getattr(cfg, "train", None)
    if train_cfg is None:
        return

    matmul_precision = getattr(train_cfg, "float32_matmul_precision", None)
    if torch.cuda.is_available() and matmul_precision:
        torch.set_float32_matmul_precision(str(matmul_precision))

    allow_tf32 = getattr(train_cfg, "allow_tf32", None)
    if allow_tf32 is not None and not deterministic:
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)


def _validate_experiment_cfg(cfg: Any) -> None:
    ref_mode = str(OmegaConf.select(cfg, "dataset.train_ref_clip_mode", default="")).strip().lower()
    ref_mode = ref_mode.replace("-", "_")
    ref_mode_aliases = {"near", "nearby_nonoverlap", "nearby_non_overlap"}
    uses_nearby_ref = ref_mode == "nearby" or ref_mode in ref_mode_aliases

    speaker_cfg = getattr(getattr(cfg, "model", None), "speaker_encoder", None)
    disables_wavlm_pos = False
    if speaker_cfg is not None:
        disables_wavlm_pos = _as_bool(getattr(speaker_cfg, "wavlm_disable_pos_conv", False)) or _as_bool(
            getattr(speaker_cfg, "wavlm_disable_relative_position_bias", False)
        )

    if disables_wavlm_pos and uses_nearby_ref:
        raise ValueError(
            "Invalid experiment config: WavLM no-position mode must not use nearby reference clips. "
            "Set dataset.train_ref_clip_mode=target_context."
        )


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _refclip_slug(cfg: Any) -> str | None:
    ref_mode = str(OmegaConf.select(cfg, "dataset.train_ref_clip_mode", default="")).strip().lower()
    ref_mode = ref_mode.replace("-", "_")
    if ref_mode in {"near", "nearby", "nearby_nonoverlap", "nearby_non_overlap"}:
        return "nearbyrefclip"
    if ref_mode in {"target_context", "same_context", "target_context_ref", "target_inclusive"}:
        return "targetctxrefclip"
    if ref_mode in {"target", "same", "target_wav"}:
        return "nogetrefclip"
    if ref_mode in {"random", "get_ref_clip", "random_crop"}:
        return "getrefclip"
    return None


def _wavlm_pos_slug(cfg: Any) -> str | None:
    disable_pos = _as_bool(OmegaConf.select(cfg, "model.speaker_encoder.wavlm_disable_pos_conv", default=False))
    disable_rel = _as_bool(
        OmegaConf.select(cfg, "model.speaker_encoder.wavlm_disable_relative_position_bias", default=False)
    )
    if disable_pos and disable_rel:
        return "no_posconv_no_relpos"
    if disable_pos:
        return "no_posconv"
    if disable_rel:
        return "no_relpos"
    return None


def _sync_explicit_wandb_name(cfg: Any) -> None:
    explicit_name = OmegaConf.select(cfg, "train.wandb_name", default=None)
    if not explicit_name:
        return

    ref_tokens = {"nearbyrefclip", "targetctxrefclip", "nogetrefclip", "getrefclip"}
    wavlm_tokens = {"wavlm_nopos", "no_posconv_no_relpos", "no_posconv", "no_relpos"}
    ref_slug = _refclip_slug(cfg)
    wavlm_slug = _wavlm_pos_slug(cfg)

    parts = [part for part in str(explicit_name).strip("/").split("/") if part]
    out: list[str] = []
    inserted_wavlm = False
    replaced_ref = False
    for part in parts:
        if part in ref_tokens and ref_slug:
            out.append(ref_slug)
            replaced_ref = True
            if wavlm_slug and not inserted_wavlm:
                out.append(wavlm_slug)
                inserted_wavlm = True
            continue
        if part in wavlm_tokens:
            continue
        out.append(part)

    if wavlm_slug and not inserted_wavlm:
        if ref_slug and not replaced_ref:
            out.append(ref_slug)
        out.append(wavlm_slug)

    with open_dict(cfg):
        cfg.train.wandb_name = "/".join(out)


def _hydra_run_dir() -> tuple[Path, str]:
    try:
        hydra_cfg = HydraConfig.get()
        run_dir = Path(hydra_cfg.runtime.output_dir)
        config_name = str(hydra_cfg.job.config_name)
    except Exception:
        run_dir = Path.cwd()
        config_name = "sdpcodec_vqw2v_rvq300"
    return run_dir.resolve(), config_name


def _timestamp_slug(run_dir: Path) -> str:
    return f"{run_dir.parent.name}-{run_dir.name}"


def _prepare_log_dir(cfg: Any, run_dir: Path) -> Path:
    log_dir = Path(str(cfg.log_dir))
    if not log_dir.is_absolute():
        log_dir = run_dir / log_dir
    with open_dict(cfg):
        cfg.log_dir = str(log_dir)
    return log_dir


def _save_resolved_config(cfg: Any, run_dir: Path, config_name: str) -> None:
    if not _is_main_process():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / f"{config_name}.yaml")
    hydra_dir = run_dir / "hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, hydra_dir / f"{config_name}.yaml")


def _wandb_name(cfg: Any, ts_slug: str) -> str:
    base_name = str(getattr(cfg.train, "wandb_name", "sdpcodec/vqw2v/rvq300")).strip("/")
    if getattr(cfg.train, "wandb_append_timestamp", True):
        return f"{base_name}/{ts_slug}"
    return base_name


def _build_logger(cfg: Any, run_dir: Path, ts_slug: str):
    enabled = bool(getattr(cfg.train, "wandb_enabled", True))
    enabled = enabled and os.environ.get("WANDB_MODE", "").lower() != "disabled"
    if not enabled:
        return False

    run_id = str(getattr(cfg, "id", None) or ts_slug)
    return WandbLogger(
        save_dir=str(run_dir / "logs"),
        name=_wandb_name(cfg, ts_slug),
        project=str(getattr(cfg.train, "wandb_project", "SDPCodec")),
        offline=bool(getattr(cfg.train, "wandb_offline", False) or os.environ.get("WANDB_MODE", "").lower() == "offline"),
        id=run_id,
        resume="allow" if getattr(cfg, "ckpt", None) else None,
    )


def _build_callbacks(cfg: Any, wandb_enabled: bool) -> list[pl.Callback]:
    callbacks: list[pl.Callback] = []
    if wandb_enabled:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    checkpointing = bool(getattr(cfg.train.trainer, "enable_checkpointing", True))
    if checkpointing:
        callbacks.insert(
            0,
            ModelCheckpoint(
                dirpath=str(cfg.log_dir),
                save_top_k=3,
                save_last=True,
                monitor="val_stats/stoi",
                mode="max",
                filename="step={val_stats/total_step:07}-stoi={val_stats/stoi:.4f}",
                auto_insert_metric_name=False,
            ),
        )
        callbacks.append(SamplerStateCheckpointCallback())
    return callbacks


def _trainer_kwargs(cfg: Any, callbacks: list[pl.Callback]) -> dict[str, Any]:
    kwargs = dict(OmegaConf.to_container(cfg.train.trainer, resolve=True))
    kwargs.setdefault("deterministic", bool(getattr(cfg, "deterministic", False)))

    if _device_count(kwargs.get("devices", 1)) > 1:
        kwargs["strategy"] = DDPStrategy(
            process_group_backend="nccl",
            find_unused_parameters=bool(getattr(cfg.train, "ddp_find_unused_parameters", True)),
            static_graph=bool(getattr(cfg.train, "ddp_static_graph", False)),
            gradient_as_bucket_view=bool(getattr(cfg.train, "ddp_gradient_as_bucket_view", False)),
        )
        if bool(kwargs.get("enable_progress_bar", True)):
            callbacks.insert(0, RankZeroProgressBar())
    return kwargs


@hydra.main(version_base=None, config_path="../configs", config_name="sdpcodec_vqw2v_rvq300")
def train(cfg: Any) -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"No device id is provided via",
        category=UserWarning,
        module=r"torch\.distributed\.distributed_c10d",
    )

    _validate_experiment_cfg(cfg)
    _sync_explicit_wandb_name(cfg)
    _configure_reproducibility(cfg)
    run_dir, config_name = _hydra_run_dir()
    ts_slug = _timestamp_slug(run_dir)
    _prepare_log_dir(cfg, run_dir)
    _save_resolved_config(cfg, run_dir, config_name)

    logger = _build_logger(cfg, run_dir, ts_slug)
    callbacks = _build_callbacks(cfg, wandb_enabled=bool(logger))
    kwargs = _trainer_kwargs(cfg, callbacks)

    datamodule = SdpCodecDataModule(cfg)
    model = SdpCodecLightningModule(cfg)
    trainer = pl.Trainer(**kwargs, callbacks=callbacks, logger=logger)
    trainer.fit(model, datamodule=datamodule, ckpt_path=getattr(cfg, "ckpt", None))


if __name__ == "__main__":
    train()
