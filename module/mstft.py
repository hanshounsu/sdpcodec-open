from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from common.audio import stft as legacy_stft


class LegacyNLayerSpecDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: Sequence[int] = (5, 3),
        channels: int = 32,
        max_downsample_channels: int = 512,
        downsample_scales: Sequence[int] = (2, 2, 2),
    ):
        super().__init__()
        kernel_sizes = tuple(int(v) for v in kernel_sizes)
        if kernel_sizes[0] % 2 != 1 or kernel_sizes[1] % 2 != 1:
            raise ValueError(f"LegacyNLayerSpecDiscriminator expects odd kernel sizes, got {kernel_sizes}.")

        model = nn.ModuleDict()
        model["layer_0"] = nn.Sequential(
            nn.Conv2d(
                int(in_channels),
                int(channels),
                kernel_size=kernel_sizes[0],
                stride=2,
                padding=kernel_sizes[0] // 2,
            ),
            nn.LeakyReLU(0.2, True),
        )

        in_chs = int(channels)
        max_downsample_channels = int(max_downsample_channels)
        for idx, downsample_scale in enumerate(tuple(int(v) for v in downsample_scales)):
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)
            model[f"layer_{idx + 1}"] = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=downsample_scale * 2 + 1,
                    stride=downsample_scale,
                    padding=downsample_scale,
                ),
                nn.LeakyReLU(0.2, True),
            )
            in_chs = out_chs

        out_chs = min(in_chs * 2, max_downsample_channels)
        model[f"layer_{len(tuple(downsample_scales)) + 1}"] = nn.Sequential(
            nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size=kernel_sizes[1],
                padding=kernel_sizes[1] // 2,
            ),
            nn.LeakyReLU(0.2, True),
        )
        model[f"layer_{len(tuple(downsample_scales)) + 2}"] = nn.Conv2d(
            out_chs,
            int(out_channels),
            kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1] // 2,
        )
        self.model = model

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        results = []
        for _, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class LegacySpecDiscriminator(nn.Module):
    def __init__(
        self,
        stft_params: dict | None = None,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: Sequence[int] = (7, 3),
        channels: int = 32,
        max_downsample_channels: int = 512,
        downsample_scales: Sequence[int] = (2, 2, 2),
        use_weight_norm: bool = True,
    ):
        super().__init__()
        if stft_params is None:
            stft_params = {
                "fft_sizes": [1024, 2048, 512],
                "hop_sizes": [120, 240, 50],
                "win_lengths": [600, 1200, 240],
                "window": "hann_window",
            }
        self.stft_params = stft_params
        self.model = nn.ModuleDict()
        for idx in range(len(stft_params["fft_sizes"])):
            self.model[f"disc_{idx}"] = LegacyNLayerSpecDiscriminator(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                channels=channels,
                max_downsample_channels=max_downsample_channels,
                downsample_scales=downsample_scales,
            )
        if use_weight_norm:
            self.apply_weight_norm()
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> list[list[torch.Tensor]]:
        results = []
        x = x.squeeze(1)
        for idx, disc in enumerate(self.model.values()):
            win_length = int(self.stft_params["win_lengths"][idx])
            spec = legacy_stft(
                x,
                int(self.stft_params["fft_sizes"][idx]),
                int(self.stft_params["hop_sizes"][idx]),
                win_length,
                window=getattr(torch, self.stft_params["window"])(win_length),
            )
            spec = spec.transpose(1, 2).unsqueeze(1)
            results.append(disc(spec))
        return results

    def remove_weight_norm(self) -> None:
        def _remove_weight_norm(module: nn.Module) -> None:
            try:
                torch.nn.utils.remove_weight_norm(module)
            except ValueError:
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self) -> None:
        def _apply_weight_norm(module: nn.Module) -> None:
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.utils.weight_norm(module)

        self.apply(_apply_weight_norm)

    def reset_parameters(self) -> None:
        def _reset_parameters(module: nn.Module) -> None:
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
                module.weight.data.normal_(0.0, 0.02)

        self.apply(_reset_parameters)
