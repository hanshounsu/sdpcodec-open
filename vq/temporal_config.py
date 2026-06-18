"""
Unified temporal backbone config for codec encoder/decoder.

YAML can use either a nested block or legacy flat keys.

**Encoder (codec_encoder)** — nested::

    temporal:
      use: true
      type: lstm
      bidirectional: false

**Decoder (codec_decoder)** — nested::

    temporal:
      use: true
      type: lstm
      bidirectional: false

Legacy flat keys remain supported via ``from_encoder_cfg`` / ``from_decoder_cfg``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Dict

import torch.nn as nn

from .module import _as_plain_dict, build_res_temporal

DEFAULT_LSTM_LAYERS = 2


@dataclass
class CodecTemporalConfig:
    """Temporal stack on the encoder path (before final conv projection)."""

    use: bool = True
    backbone: str = "lstm"
    bidirectional: bool = False

    def effective_num_layers(self) -> int:
        return DEFAULT_LSTM_LAYERS

    def build(self, channels: int) -> nn.Module:
        """Build ResLSTM for ``channels`` feature width."""
        return build_res_temporal(
            channels,
            self.backbone,
            num_layers=self.effective_num_layers(),
            bidirectional=self.bidirectional,
        )

    @classmethod
    def from_encoder_cfg(cls, cfg: Any) -> CodecTemporalConfig:
        """Parse ``codec_encoder`` (Hydra dict / OmegaConf / plain dict)."""
        d = _as_plain_dict(cfg)
        sub = d.get("temporal")
        if sub is not None:
            t = _as_plain_dict(sub)
            bb = t.get("type", t.get("backbone", "lstm"))
            return cls(
                use=bool(t.get("use", True)),
                backbone=str(bb).lower(),
                bidirectional=bool(t.get("bidirectional", False)),
            )
        return cls(
            use=bool(d.get("use_rnn", True)),
            backbone=str(d.get("encoder_rnn_type", "lstm")).lower(),
            bidirectional=bool(d.get("rnn_bidirectional", False)),
        )


@dataclass
class CodecDecoderTemporalConfig:
    """Temporal stack on the default decoder path."""

    use: bool = True
    backbone: str = "lstm"
    bidirectional: bool = False
    start_layer: int = 0
    end_layer: int | None = None
    every: int = 1

    def effective_num_layers(self) -> int:
        return DEFAULT_LSTM_LAYERS

    def build_res_path(self, channels: int) -> nn.Module:
        """Build the LSTM temporal path."""
        return build_res_temporal(
            channels,
            self.backbone,
            num_layers=self.effective_num_layers(),
            bidirectional=self.bidirectional,
        )

    @classmethod
    def from_decoder_cfg(cls, cfg: Any) -> CodecDecoderTemporalConfig:
        """Parse ``codec_decoder`` (Hydra dict / OmegaConf / plain dict)."""
        d = _as_plain_dict(cfg)
        sub = d.get("temporal")
        if sub is not None:
            t = _as_plain_dict(sub)
            bb = t.get("type", t.get("backbone", "lstm"))
            return cls(
                use=bool(t.get("use", True)),
                backbone=str(bb).lower(),
                bidirectional=bool(t.get("bidirectional", False)),
                start_layer=int(t.get("start_layer", 0) or 0),
                end_layer=t.get("end_layer", None),
                every=int(t.get("every", 1) or 1),
            )
        resolved = d.get("decoder_rnn_type") or d.get("rnn_type") or "lstm"
        return cls(
            use=bool(d.get("use_rnn", True)),
            backbone=str(resolved).lower(),
            bidirectional=bool(d.get("rnn_bidirectional", False)),
            start_layer=int(d.get("start_layer", 0) or 0),
            end_layer=d.get("end_layer", None),
            every=int(d.get("every", 1) or 1),
        )


@dataclass
class CodecDecoderSpeakerConditionConfig:
    use: bool = False
    type: tuple[str, ...] = ("mhca",)  # mhca | film | concat
    num_heads: int = 2
    dropout: float = 0.2
    start_layer: int = 0
    end_layer: int | None = None
    every: int = 1
    film_start_layer: int | None = None
    film_end_layer: int | None = None
    film_every: int | None = None
    mhca_start_layer: int | None = None
    mhca_end_layer: int | None = None
    mhca_every: int | None = None

    @property
    def use_mhca(self) -> bool:
        return self.use and "mhca" in self.type

    @property
    def use_concat(self) -> bool:
        return self.use and "concat" in self.type

    @property
    def use_film(self) -> bool:
        return self.use and "film" in self.type

    @staticmethod
    def _normalize_type(raw: Any) -> tuple[str, ...]:
        if raw is None:
            return ("mhca",)
        if isinstance(raw, str):
            value = raw.strip().lower()
            if value == "all":
                return ("mhca", "film")
            if value in {"mhca", "film", "concat"}:
                return (value,)
            raise ValueError(f"Unsupported codec_decoder.spk_cond.type: {value}")
        if isinstance(raw, Iterable):
            resolved: list[str] = []
            for item in raw:
                value = str(item).strip().lower()
                if value == "all":
                    value = "mhca+film"
                if value == "mhca+film":
                    for sub_value in ("mhca", "film"):
                        if sub_value not in resolved:
                            resolved.append(sub_value)
                    continue
                if value not in {"mhca", "film", "concat"}:
                    raise ValueError(f"Unsupported codec_decoder.spk_cond.type item: {value}")
                if value not in resolved:
                    resolved.append(value)
            return tuple(resolved)
        raise ValueError(f"Unsupported codec_decoder.spk_cond.type: {raw}")

    def resolve_block_range(self, total_layers: int) -> tuple[int, int, int]:
        start = max(0, min(int(self.start_layer), total_layers))
        if self.end_layer is None:
            end = total_layers
        else:
            end = max(start, min(int(self.end_layer), total_layers))
        every = max(1, int(self.every))
        return start, end, every

    @property
    def resolved_film_start_layer(self) -> int:
        return self.start_layer if self.film_start_layer is None else int(self.film_start_layer)

    @property
    def resolved_film_end_layer(self) -> int | None:
        return self.end_layer if self.film_end_layer is None else self.film_end_layer

    @property
    def resolved_film_every(self) -> int:
        return self.every if self.film_every is None else max(1, int(self.film_every))

    @property
    def resolved_mhca_start_layer(self) -> int:
        return self.start_layer if self.mhca_start_layer is None else int(self.mhca_start_layer)

    @property
    def resolved_mhca_end_layer(self) -> int | None:
        return self.end_layer if self.mhca_end_layer is None else self.mhca_end_layer

    @property
    def resolved_mhca_every(self) -> int:
        return self.every if self.mhca_every is None else max(1, int(self.mhca_every))

    @classmethod
    def from_decoder_cfg(cls, cfg: Any) -> "CodecDecoderSpeakerConditionConfig":
        d = _as_plain_dict(cfg)
        sub = _as_plain_dict(d.get("spk_cond"))
        legacy = _as_plain_dict(d.get("mhca"))
        raw_type = sub.get("type", legacy.get("type", "all"))
        return cls(
            use=bool(sub.get("use", legacy.get("use", d.get("use_mhca", False)))),
            type=cls._normalize_type(raw_type),
            num_heads=int(sub.get("num_heads", legacy.get("num_heads", d.get("mhca_num_heads", 2)))),
            dropout=float(sub.get("dropout", legacy.get("dropout", d.get("mhca_dropout", 0.2)))),
            start_layer=int(sub.get("start_layer", legacy.get("start_layer", d.get("mhca_start_layer", 0)) or 0)),
            end_layer=sub.get("end_layer", legacy.get("end_layer", d.get("mhca_end_layer", None))),
            every=int(sub.get("every", legacy.get("every", d.get("mhca_every", 1)) or 1)),
            film_start_layer=sub.get("film_start_layer", None),
            film_end_layer=sub.get("film_end_layer", None),
            film_every=sub.get("film_every", None),
            mhca_start_layer=sub.get("mhca_start_layer", legacy.get("start_layer", d.get("mhca_start_layer", None))),
            mhca_end_layer=sub.get("mhca_end_layer", legacy.get("end_layer", d.get("mhca_end_layer", None))),
            mhca_every=sub.get("mhca_every", legacy.get("every", d.get("mhca_every", None))),
        )


@dataclass
class CodecDecoderF0ConditionConfig:
    use: bool = False
    type: tuple[str, ...] = ("concat",)
    start_layer: int = 0
    end_layer: int | None = None
    every: int = 1

    @property
    def use_concat(self) -> bool:
        return self.use and "concat" in self.type

    @staticmethod
    def _normalize_type(raw: Any) -> tuple[str, ...]:
        if raw is None:
            return ("concat",)
        if isinstance(raw, str):
            value = raw.strip().lower()
            if value == "all":
                return ("concat",)
            if value == "concat":
                return ("concat",)
            raise ValueError(f"Unsupported codec_decoder.f0_condition.type: {value}")
        if isinstance(raw, Iterable):
            resolved: list[str] = []
            for item in raw:
                value = str(item).strip().lower()
                if value == "all":
                    value = "concat"
                if value != "concat":
                    raise ValueError(f"Unsupported codec_decoder.f0_condition.type item: {value}")
                if value not in resolved:
                    resolved.append(value)
            if not resolved:
                raise ValueError("codec_decoder.f0_condition.type list must not be empty.")
            return tuple(resolved)
        raise ValueError(f"Unsupported codec_decoder.f0_condition.type: {raw}")

    def resolve_block_range(self, total_layers: int) -> tuple[int, int, int]:
        start = max(0, min(int(self.start_layer), total_layers))
        if self.end_layer is None:
            end = total_layers
        else:
            end = max(start, min(int(self.end_layer), total_layers))
        every = max(1, int(self.every))
        return start, end, every

    @classmethod
    def from_decoder_cfg(cls, cfg: Any) -> "CodecDecoderF0ConditionConfig":
        d = _as_plain_dict(cfg)
        raw = d.get("f0_condition", None)
        sub = _as_plain_dict(raw)
        if sub:
            return cls(
                use=bool(sub.get("use", True)),
                type=cls._normalize_type(sub.get("type", "concat")),
                start_layer=int(sub.get("start_layer", 0) or 0),
                end_layer=sub.get("end_layer", None),
                every=int(sub.get("every", 1) or 1),
            )
        if raw is None:
            return cls(use=False, type=("concat",))
        use = bool(raw)
        return cls(use=use, type=("concat",), start_layer=0, end_layer=None, every=1)


@dataclass
class F0CodecDecoderConfig:
    """
    Unified F0 decoder config for the codec-structure F0 path.

    Supports both:
    - legacy flat keys such as ``decoder_rnn_type`` / ``decoder_use_mhca``
    - a nested ``decoder:`` block under ``f0_codec``

    Preferred nested shape::

        f0_codec:
          decoder:
            use: true
            type: lstm
            bidirectional: false
            num_layers: 2
            ngf: 16
            up_ratios: [3, 1, 1, 1]
            dilations: [1, 3, 9]
            activation_type: LeakyReLU
            leaky_relu_params:
              negative_slope: 0.1
            mhca:
              use: false
              num_heads: 2
              dropout: 0.2
              key_dim: 128
    """

    use: bool = True
    backbone: str = "lstm"
    bidirectional: bool = False
    num_layers: int = DEFAULT_LSTM_LAYERS
    ngf: int = 16
    up_ratios: tuple[Any, ...] = field(default_factory=tuple)
    dilations: tuple[Any, ...] = field(default_factory=tuple)
    activation_type: str = "LeakyReLU"
    leaky_relu_params: Dict[str, Any] | None = None
    use_mhca: bool = False
    mhca_num_heads: int = 2
    mhca_dropout: float = 0.2
    mhca_key_dim: int | None = None

    @classmethod
    def from_f0_cfg(cls, cfg: Any) -> "F0CodecDecoderConfig":
        d = _as_plain_dict(cfg)
        sub = _as_plain_dict(d.get("decoder"))
        temporal = _as_plain_dict(sub.get("temporal"))
        mhca = _as_plain_dict(sub.get("mhca"))

        backbone = temporal.get(
            "type",
            temporal.get(
                "backbone",
                sub.get("type", sub.get("backbone", d.get("decoder_rnn_type", "lstm"))),
            ),
        )

        use = temporal.get("use", sub.get("use", d.get("decoder_use_rnn", True)))
        bidirectional = temporal.get(
            "bidirectional",
            sub.get("bidirectional", d.get("decoder_rnn_bidirectional", False)),
        )
        num_layers = temporal.get(
            "num_layers",
            sub.get("num_layers", d.get("decoder_rnn_num_layers", DEFAULT_LSTM_LAYERS)),
        )

        return cls(
            use=bool(use),
            backbone=str(backbone).lower(),
            bidirectional=bool(bidirectional),
            num_layers=int(num_layers),
            ngf=int(sub.get("ngf", d.get("decoder_ngf", 16))),
            up_ratios=tuple(sub.get("up_ratios", d.get("decoder_up_ratios", ()))),
            dilations=tuple(sub.get("dilations", d.get("decoder_dilations", ()))),
            activation_type=str(sub.get("activation_type", d.get("decoder_activation_type", "LeakyReLU"))),
            leaky_relu_params=sub.get(
                "leaky_relu_params",
                d.get("decoder_leaky_relu_params", None),
            ),
            use_mhca=bool(mhca.get("use", sub.get("use_mhca", d.get("decoder_use_mhca", d.get("use_mhca", False))))),
            mhca_num_heads=int(mhca.get("num_heads", sub.get("mhca_num_heads", d.get("decoder_mhca_num_heads", 2)))),
            mhca_dropout=float(mhca.get("dropout", sub.get("mhca_dropout", d.get("decoder_mhca_dropout", 0.2)))),
            mhca_key_dim=mhca.get("key_dim", sub.get("mhca_key_dim", d.get("decoder_mhca_key_dim", None))),
        )


@dataclass
class F0CodecEncoderConfig:
    """
    Unified F0 encoder config for the codec-structure F0 path.

    Supports both:
    - legacy flat keys such as ``encoder_use_rnn``
    - a nested ``encoder:`` block under ``f0_codec``
    """

    use: bool = True
    backbone: str = "lstm"
    bidirectional: bool = False
    num_layers: int = DEFAULT_LSTM_LAYERS
    ngf: int = 16
    up_ratios: tuple[Any, ...] = field(default_factory=tuple)
    dilations: tuple[Any, ...] = field(default_factory=tuple)
    out_channels: int = 128
    activation_type: str = "LeakyReLU"
    leaky_relu_params: Dict[str, Any] | None = None

    @classmethod
    def from_f0_cfg(cls, cfg: Any) -> "F0CodecEncoderConfig":
        d = _as_plain_dict(cfg)
        sub = _as_plain_dict(d.get("encoder"))
        temporal = _as_plain_dict(sub.get("temporal"))

        backbone = temporal.get(
            "type",
            temporal.get(
                "backbone",
                sub.get("type", sub.get("backbone", d.get("encoder_rnn_type", "lstm"))),
            ),
        )
        use = temporal.get("use", sub.get("use", d.get("encoder_use_rnn", True)))
        bidirectional = temporal.get(
            "bidirectional",
            sub.get("bidirectional", d.get("encoder_rnn_bidirectional", False)),
        )
        num_layers = temporal.get(
            "num_layers",
            sub.get("num_layers", d.get("encoder_rnn_num_layers", DEFAULT_LSTM_LAYERS)),
        )

        return cls(
            use=bool(use),
            backbone=str(backbone).lower(),
            bidirectional=bool(bidirectional),
            num_layers=int(num_layers),
            ngf=int(sub.get("ngf", d.get("encoder_ngf", 16))),
            up_ratios=tuple(sub.get("up_ratios", d.get("encoder_up_ratios", ()))),
            dilations=tuple(sub.get("dilations", d.get("encoder_dilations", ()))),
            out_channels=int(sub.get("out_channels", d.get("encoder_out_channels", 128))),
            activation_type=str(sub.get("activation_type", d.get("encoder_activation_type", "LeakyReLU"))),
            leaky_relu_params=sub.get(
                "leaky_relu_params",
                d.get("encoder_leaky_relu_params", None),
            ),
        )


@dataclass
class F0CodecSpeakerConditionConfig:
    """
    Speaker-conditioning policy for the codec-structure F0 decoder.

    ``f0_codec.spk_cond`` controls whether the F0 decoder uses:
    - ``concat``: global speaker embedding concatenation before each decoder block
    - ``mhca``: cross-attention over a time-varying speaker sequence
    - ``film``: per-stage FiLM modulation from the global speaker embedding

    Backward compatibility:
    - If ``spk_cond.use`` is omitted, legacy auto-enable logic is preserved.
    - If ``spk_cond.type`` is omitted, legacy ``decoder_use_mhca`` maps to
      ``[mhca, concat]`` and otherwise defaults to ``[concat]``.
    """

    use: bool | None = None
    type: tuple[str, ...] = ("concat",)
    num_heads: int = 2
    dropout: float = 0.2
    key_dim: int | None = None

    @property
    def has_explicit_use(self) -> bool:
        return self.use is not None

    @staticmethod
    def _normalize_type(raw: Any) -> tuple[str, ...]:
        if raw is None:
            return ("concat",)
        if isinstance(raw, str):
            value = raw.strip().lower()
            if value == "all":
                return ("mhca", "concat", "film")
            if value in {"mhca", "concat", "film"}:
                return (value,)
            raise ValueError(f"Unsupported f0_codec.spk_cond.type: {value}")
        if isinstance(raw, Iterable):
            resolved: list[str] = []
            for item in raw:
                value = str(item).strip().lower()
                if value == "all":
                    value = "mhca+concat+film"
                if value == "mhca+concat+film":
                    for sub_value in ("mhca", "concat", "film"):
                        if sub_value not in resolved:
                            resolved.append(sub_value)
                    continue
                if value not in {"mhca", "concat", "film"}:
                    raise ValueError(f"Unsupported f0_codec.spk_cond.type item: {value}")
                if value not in resolved:
                    resolved.append(value)
            if not resolved:
                raise ValueError("f0_codec.spk_cond.type list must not be empty.")
            return tuple(resolved)
        raise ValueError(f"Unsupported f0_codec.spk_cond.type: {raw}")

    def resolve_enabled(self, default_use: bool) -> bool:
        return default_use if self.use is None else bool(self.use)

    def resolve_concat(self, default_use: bool) -> bool:
        return self.resolve_enabled(default_use) and "concat" in self.type

    def resolve_film(self, default_use: bool) -> bool:
        return self.resolve_enabled(default_use) and "film" in self.type

    def resolve_mhca(self, default_use: bool) -> bool:
        return self.resolve_enabled(default_use) and "mhca" in self.type

    @classmethod
    def from_f0_cfg(cls, cfg: Any) -> "F0CodecSpeakerConditionConfig":
        d = _as_plain_dict(cfg)
        sub = _as_plain_dict(d.get("spk_cond"))
        decoder_sub = _as_plain_dict(d.get("decoder"))
        decoder_mhca = _as_plain_dict(decoder_sub.get("mhca"))

        legacy_use_mhca = bool(d.get("decoder_use_mhca", d.get("use_mhca", False)))
        if "type" in sub:
            raw_type = sub.get("type")
        else:
            raw_type = ("mhca", "concat") if (
                decoder_mhca.get("use", legacy_use_mhca) or decoder_sub.get("use_mhca", False)
            ) else ("concat",)

        explicit_use = sub.get("use") if "use" in sub else None
        return cls(
            use=None if explicit_use is None else bool(explicit_use),
            type=cls._normalize_type(raw_type),
            num_heads=int(sub.get("num_heads", decoder_mhca.get("num_heads", d.get("decoder_mhca_num_heads", 2)))),
            dropout=float(sub.get("dropout", decoder_mhca.get("dropout", d.get("decoder_mhca_dropout", 0.2)))),
            key_dim=sub.get("key_dim", decoder_mhca.get("key_dim", d.get("decoder_mhca_key_dim", None))),
        )
