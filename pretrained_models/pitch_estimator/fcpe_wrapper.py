from __future__ import annotations

import sys
from pathlib import Path

import torch


_FCPE_ROOT = Path(__file__).resolve().parent / "FCPE"
_FCPE_PATCHED = False


def _ensure_fcpe_root_on_path() -> None:
    if not (_FCPE_ROOT / "torchfcpe").is_dir():
        raise FileNotFoundError(
            "FCPE submodule is missing. Run: git submodule update --init --recursive"
        )
    if str(_FCPE_ROOT) not in sys.path:
        sys.path.insert(0, str(_FCPE_ROOT))


def _load_fcpe_modules():
    _ensure_fcpe_root_on_path()
    try:
        from torchfcpe.models import CFNaiveMelPE
        from torchfcpe.models_infer import spawn_bundled_infer_model, spawn_model
        from torchfcpe.tools import DotDict, spawn_wav2mel
    except ModuleNotFoundError as exc:
        if exc.name == "local_attention":
            raise ModuleNotFoundError(
                "FCPE requires local_attention. Install repository requirements first: "
                "pip install -r requirements.txt"
            ) from exc
        raise
    return CFNaiveMelPE, spawn_bundled_infer_model, spawn_model, DotDict, spawn_wav2mel


def _patch_fcpe_model_decoders(CFNaiveMelPE) -> None:
    global _FCPE_PATCHED
    if _FCPE_PATCHED:
        return

    @torch.no_grad()
    def latent2cents_decoder(self, y, threshold=0.05, mask=True):
        batch_size, length, _ = y.size()
        ci = self.cent_table[None, None, :].expand(batch_size, length, -1)
        cents = torch.sum(ci * y, dim=-1, keepdim=True) / torch.sum(y, dim=-1, keepdim=True)
        if mask:
            confident = torch.max(y, dim=-1, keepdim=True)[0]
            cents = cents.masked_fill(confident <= threshold, float("-INF"))
        return cents

    @torch.no_grad()
    def latent2cents_local_decoder(self, y, threshold=0.05, mask=True):
        batch_size, length, _ = y.size()
        ci = self.cent_table[None, None, :].expand(batch_size, length, -1)
        confident, max_index = torch.max(y, dim=-1, keepdim=True)
        local_argmax_index = torch.arange(0, 9).to(max_index.device) + (max_index - 4)
        local_argmax_index = torch.clamp(local_argmax_index, min=0)
        local_argmax_index = local_argmax_index.clamp(0, self.out_dims - 1)
        ci_l = torch.gather(ci, -1, local_argmax_index)
        y_l = torch.gather(y, -1, local_argmax_index)
        cents = torch.sum(ci_l * y_l, dim=-1, keepdim=True) / torch.sum(y_l, dim=-1, keepdim=True)
        if mask:
            cents = cents.masked_fill(confident <= threshold, float("-INF"))
        return cents

    CFNaiveMelPE.latent2cents_decoder = latent2cents_decoder
    CFNaiveMelPE.latent2cents_local_decoder = latent2cents_local_decoder
    _FCPE_PATCHED = True


class F0ExtractorWrapper(torch.nn.Module):
    def __init__(self, cfg, device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device

        from termcolor import colored

        (
            CFNaiveMelPE,
            spawn_bundled_infer_model,
            spawn_model,
            DotDict,
            spawn_wav2mel,
        ) = _load_fcpe_modules()
        _patch_fcpe_model_decoders(CFNaiveMelPE)

        if not cfg.model.f0_codec.finetune_f0_extractor:
            print(colored("Use pretrained F0 extractor WITHOUT finetuning", "red"))
            self.extractor = spawn_bundled_infer_model(device=device)
            for param in self.extractor.parameters():
                param.requires_grad = False
            self.extractor.eval()
            self.is_infer = True
        else:
            ckpt_path = _FCPE_ROOT / "torchfcpe" / "assets" / "fcpe_c_v001.pt"
            ckpt = torch.load(ckpt_path, map_location=device)
            print(colored(f"Use pretrained F0 extractor WITH finetuning, ckpt: {ckpt_path}", "red"))
            args = DotDict(ckpt["config_dict"])
            self.wav2mel = spawn_wav2mel(args, device=device)
            self.model = spawn_model(args)
            self.model.load_state_dict(ckpt["model"])
            self.model = self.model.to(device)
            self.model.train()
            self.is_infer = False

    def forward(
        self,
        wav,
        sr=16000,
        decoder_mode="local_argmax",
        threshold=0.006,
        f0_min=80.0,
        f0_max=880.0,
        interp_uv=False,
        output_interp_target_length=None,
        retur_uv=False,
        **kwargs,
    ):
        if self.is_infer:
            with torch.no_grad():
                return self.extractor.infer(
                    wav,
                    sr=sr,
                    decoder_mode=decoder_mode,
                    threshold=threshold,
                    f0_min=f0_min,
                    f0_max=f0_max,
                    interp_uv=interp_uv,
                    output_interp_target_length=output_interp_target_length,
                    retur_uv=retur_uv,
                    **kwargs,
                ).squeeze(-1).float()

        mel = self.wav2mel(wav, sr)
        latents = self.model(mel, **kwargs)
        if decoder_mode == "argmax":
            cents = self.model.latent2cents_decoder(latents, threshold=threshold)
        elif decoder_mode == "local_argmax":
            cents = self.model.latent2cents_local_decoder(latents, threshold=threshold)
        else:
            raise ValueError(f"Unknown decoder_mode: {decoder_mode}")

        f0 = self.model.cent_to_f0(cents)
        if f0_min is None:
            f0_min = self.model.f0_min
        uv = (f0 < f0_min).type(f0.dtype)
        if interp_uv:
            _ensure_fcpe_root_on_path()
            from torchfcpe.torch_interp import batch_interp_with_replacement_detach

            f0 = batch_interp_with_replacement_detach(
                uv.squeeze(-1).bool(),
                f0.squeeze(-1),
            ).unsqueeze(-1)
        if f0_max is not None:
            f0 = torch.clamp(f0, max=f0_max)
        if output_interp_target_length is not None:
            f0 = torch.nn.functional.interpolate(
                f0.transpose(1, 2),
                size=int(output_interp_target_length),
                mode="nearest",
            ).transpose(1, 2)
        if retur_uv:
            uv = torch.nn.functional.interpolate(
                uv.transpose(1, 2),
                size=int(output_interp_target_length),
                mode="nearest",
            ).transpose(1, 2)
            return f0.squeeze(-1).float(), uv.squeeze(-1).float()
        return f0.squeeze(-1).float()

    def f0_to_cent(self, f0):
        if self.is_infer:
            return self.extractor.model.f0_to_cent(f0)
        return self.model.f0_to_cent(f0)

    def gaussian_blurred_cent2latent(self, cent):
        if self.is_infer:
            return self.extractor.model.gaussian_blurred_cent2latent(cent)
        return self.model.gaussian_blurred_cent2latent(cent)
