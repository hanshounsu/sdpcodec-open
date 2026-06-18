"""Single-file SDPCodec inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
from omegaconf import OmegaConf, open_dict

from sdpcodec.system import SdpCodecLightningModule


def _load_mono(path: Path, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.float()
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if int(sr) != int(sample_rate):
        wav = torchaudio.functional.resample(wav, int(sr), int(sample_rate))
    return wav.squeeze(0).clamp(-1.0, 1.0)


def _align_length(wav: torch.Tensor, hop_length: int, min_length: int = 0) -> torch.Tensor:
    if min_length > 0 and wav.numel() < min_length:
        wav = F.pad(wav, (0, min_length - wav.numel()))
    if hop_length > 0:
        frames = max(1, wav.numel() // hop_length)
        wav = wav[: frames * hop_length]
    return wav


def _prepare_ref(wav: torch.Tensor, sample_rate: int, seconds: float, hop_length: int) -> torch.Tensor:
    target_len = int(float(seconds) * int(sample_rate))
    if hop_length > 0:
        target_len = max(hop_length, (target_len // hop_length) * hop_length)
    if wav.numel() < target_len:
        repeats = (target_len + wav.numel() - 1) // max(1, wav.numel())
        wav = wav.repeat(repeats)
    return wav[:target_len].clamp(-1.0, 1.0)


def _load_model(cfg: Any, ckpt_path: Path, device: torch.device, strict: bool) -> SdpCodecLightningModule:
    model = SdpCodecLightningModule(cfg)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
    model.eval().to(device)
    model._remove_all_weight_norms()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SDPCodec inference on one waveform.")
    parser.add_argument("--checkpoint", "--ckpt", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path, help="Source wav to reconstruct or convert.")
    parser.add_argument("--reference", type=Path, default=None, help="Reference speaker wav. Defaults to source.")
    parser.add_argument("--output", required=True, type=Path, help="Output wav path.")
    parser.add_argument("--config", type=Path, default=Path("configs/sdpcodec_vqw2v_rvq300.yaml"))
    parser.add_argument("--mode", choices=["rec", "vc"], default="rec")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strict", action="store_true", help="Use strict checkpoint loading.")
    parser.add_argument("--save-codes", type=Path, default=None, help="Optional .pt path for RVQ codes.")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    with open_dict(cfg):
        cfg.ckpt = str(args.checkpoint)
        cfg.train.wandb_enabled = False
        cfg.train.use_val_utmos = False
        cfg.train.use_val_wer = False

    sample_rate = int(cfg.preprocess.audio.sr)
    hop_length = int(cfg.dataset.latent_hop_length)
    min_length = int(cfg.dataset.min_audio_length)
    device = torch.device(args.device)

    source = _load_mono(args.source, sample_rate)
    source = _align_length(source, hop_length, min_length=min_length)
    ref_path = args.reference if args.reference is not None else args.source
    ref = _load_mono(ref_path, 16000)
    ref_hop_length = hop_length if sample_rate == 16000 else 320
    ref = _prepare_ref(ref, 16000, float(cfg.dataset.ref_segment_duration), ref_hop_length)

    batch = {
        "wav": source.unsqueeze(0).to(device),
        "wav_24k": None,
        "ref_wav": ref.unsqueeze(0).to(device),
    }
    if sample_rate != 16000:
        source_16k = torchaudio.functional.resample(source.unsqueeze(0), sample_rate, 16000).squeeze(0)
        batch["wav"] = _align_length(source_16k, 320).unsqueeze(0).to(device)
        batch["wav_24k"] = source.unsqueeze(0).to(device)

    model = _load_model(cfg, args.checkpoint, device=device, strict=bool(args.strict))
    with torch.inference_mode():
        output = model(batch)

    wav = output["gen_wav"][0].detach().cpu().float().clamp(-1.0, 1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output), wav, sample_rate)
    if args.save_codes is not None:
        args.save_codes.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output["vq_code"].detach().cpu(), args.save_codes)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
