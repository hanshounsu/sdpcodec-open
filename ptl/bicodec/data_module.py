from functools import wraps
import time
import logging
from copy import deepcopy
import os

# Configure HF timeouts before any hub/datasets import so the libraries pick them up.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(120))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(120))

from termcolor import colored
from typing import Any, Dict, Iterator, List, Optional
import torch.distributed as dist
from ptl.sampler import ResumableRandomSampler, ResumableDistributedSampler
from datasets import load_dataset, concatenate_datasets, DownloadConfig
import hydra
from torch.utils.data import Dataset, DataLoader, Sampler, IterableDataset, get_worker_info
import librosa
import random
import pytorch_lightning as pl
import torchaudio
import torch.nn.functional as F
import torch
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def _original_cwd() -> str:
    try:
        return hydra.utils.get_original_cwd()
    except Exception:
        return os.getcwd()


def _vctk_keep_mic1(ex: Dict[str, Any]) -> bool:
    """Keep only mic1; mic2 is never used for VCTK eval."""
    sid = str(ex.get("id", ""))
    return sid.endswith("_mic1.flac") or sid.endswith("_mic1.wav")


def _vctk_shorten_id(ex: Dict[str, Any]) -> Dict[str, Any]:
    """Shorten id: /path/to/p225_001_mic1.flac -> p225_001_mic1."""
    sid = str(ex.get("id", ""))
    stem = os.path.splitext(os.path.basename(sid))[0]
    return {"id": stem}


def _get_hf_token() -> Optional[str]:
    """
    Return a Hugging Face token from common environment variables.

    We prefer env-based tokens because Vast.ai often runs on shared public IPs
    that hit anonymous rate limits quickly.
    """
    tok = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if tok is None:
        return None
    tok = str(tok).strip()
    return tok or None


def _load_dataset_with_token(*args: Any, **kwargs: Any) -> Any:
    """
    Wrapper around `datasets.load_dataset` that automatically injects a HF token
    if present.

    Supports both newer `token=...` and older `use_auth_token=...` APIs.
    """
    def _dataset_cache_root() -> str:
        cache_dir = kwargs.get("cache_dir")
        if cache_dir:
            return str(cache_dir)
        hf_datasets_cache = os.environ.get("HF_DATASETS_CACHE")
        if hf_datasets_cache:
            return str(hf_datasets_cache)
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        return str(os.path.join(hf_home, "datasets"))

    def _has_local_dataset_cache() -> bool:
        if not args:
            return False
        repo_id = str(args[0])
        if "/" not in repo_id:
            return False
        dataset_dir = Path(_dataset_cache_root()) / repo_id.replace("/", "___")
        return dataset_dir.exists()

    def _looks_like_hf_network_timeout(err: Exception) -> bool:
        text = f"{type(err).__name__}: {err}"
        needles = (
            "ReadTimeout",
            "Read timed out",
            "ConnectionError",
            "ConnectTimeout",
            "Temporary failure in name resolution",
            "MaxRetryError",
            "HTTPSConnectionPool",
        )
        return any(needle in text for needle in needles)

    def _make_offline_kwargs(src_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        offline_kwargs = dict(src_kwargs)
        dc = offline_kwargs.get("download_config")
        if dc is not None:
            try:
                dc = deepcopy(dc)
            except Exception:
                dc = None
        if dc is None:
            dc = DownloadConfig()
        dc.local_files_only = True
        dc.max_retries = 1
        offline_kwargs["download_config"] = dc
        return offline_kwargs

    def _load_with_retry(load_kwargs: Dict[str, Any]) -> Any:
        try:
            return load_dataset(*args, **load_kwargs)
        except Exception as err:
            if not (_looks_like_hf_network_timeout(err) and _has_local_dataset_cache()):
                raise
            logger.warning(
                "HF dataset metadata request timed out for %s. Falling back to local cached files only.",
                args[0] if args else "<unknown>",
            )
            return load_dataset(*args, **_make_offline_kwargs(load_kwargs))

    # If caller already specified auth, respect it.
    if "token" in kwargs or "use_auth_token" in kwargs:
        return _load_with_retry(kwargs)

    tok = _get_hf_token()
    if not tok:
        return _load_with_retry(kwargs)

    # Newer datasets/huggingface_hub
    try:
        return _load_with_retry({**kwargs, "token": tok})
    except TypeError:
        # Older datasets API
        return _load_with_retry({**kwargs, "use_auth_token": tok})


def _dist_rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank()), int(dist.get_world_size())
    return 0, 1


def _shard_hf_iterable(ds: Any, rank: int, world: int, worker_id: int, num_workers: int) -> Any:
    """
    Best-effort sharding for HF streaming IterableDataset across:
    - DDP rank (node)
    - DataLoader workers
    """
    if hasattr(ds, "split_by_node"):
        ds = ds.split_by_node(rank=rank, world_size=world)
    if hasattr(ds, "split_by_worker"):
        return ds.split_by_worker(worker_id=worker_id, num_workers=num_workers)

    # Fallback: modulo sharding on the iterator index
    num_shards = int(world) * int(num_workers)
    shard_index = int(rank) * int(num_workers) + int(worker_id)
    if hasattr(ds, "shard"):
        return ds.shard(num_shards=num_shards, index=shard_index)
    return (ex for i, ex in enumerate(ds) if (i % num_shards) == shard_index)


class MlsStreamingTrainDataset(IterableDataset):
    """
    Streaming MLS loader for SDPCodec training.

    IMPORTANT:
    - This is IterableDataset (no __len__/random access), so it cannot support
      VC-pair mapping logic used for val/test in this module.
    - Intended for train split only.
    """

    def __init__(
        self,
        cfg: Any,
        dc: DownloadConfig,
        *,
        dataset_id: str,
        languages: List[str],
        split: str,
        shuffle: bool,
        shuffle_buffer_size: int,
        seed: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.dc = dc
        self.dataset_id = str(dataset_id)
        self.languages = [str(x).lower() for x in (languages or [])]
        self.split = str(split)
        self.shuffle = bool(shuffle)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.seed = int(seed)

        # target sample-rate used by downstream code
        self.target_sr = int(getattr(getattr(cfg, "preprocess", None), "audio", None).sr) if hasattr(getattr(cfg, "preprocess", None), "audio") else int(getattr(cfg.dataset, "sample_rate", 16000))
        self.min_audio_length = int(getattr(cfg.dataset, "min_audio_length", 64000))

        # NOTE:
        # We intentionally keep the shuffle mechanism identical to `hoyeol/`:
        # - shard first (DDP + DataLoader workers)
        # - then call HF streaming `.shuffle(buffer_size=..., seed=...)`
        # In HF streaming, shuffle warms up its internal buffer before yielding the first sample.

    def _load_hf_iterable(self) -> Any:
        from datasets import Audio, interleave_datasets

        cache_dir = _resolve_hf_datasets_cache_dir(self.cfg)

        def _load_one(lang: Optional[str]) -> Any:
            if lang is None:
                return _load_dataset_with_token(
                    self.dataset_id,
                    split=self.split,
                    streaming=True,
                    cache_dir=cache_dir,
                    download_config=self.dc,
                )
            return _load_dataset_with_token(
                self.dataset_id,
                lang,
                split=self.split,
                streaming=True,
                cache_dir=cache_dir,
                download_config=self.dc,
            )

        # English-only MLS (parler-tts/mls_eng) has no language config.
        if self.dataset_id == "parler-tts/mls_eng":
            ds = _load_one(None)
        else:
            if len(self.languages) == 0:
                raise ValueError(
                    "cfg.dataset.mls.languages must be non-empty for streaming multilingual MLS."
                )
            dsets = [_load_one(lang) for lang in self.languages]
            ds = dsets[0] if len(dsets) == 1 else interleave_datasets(dsets, seed=self.seed)

        # Ensure decoded audio arrays in streaming mode.
        ds = ds.cast_column("audio", Audio(decode=True))
        return ds

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        import torch
        import torchaudio

        def _decode_audio_to_wav_sr(audio_obj: Any) -> Optional[tuple[torch.Tensor, int]]:
            """
            Minimal HF Audio handling (same idea as `hoyeol/`):
            - Prefer decoded waveform (audio["array"], audio["sampling_rate"])
            - Fallback to audio["bytes"] if present (decode ourselves)

            NOTE: We intentionally do NOT implement a path-based loader here to keep the logic simple.
            """
            if audio_obj is None:
                return None

            def _get_field(obj: Any, key: str) -> Any:
                if isinstance(obj, dict):
                    return obj.get(key)
                # Some HF audio objects are subscriptable but not Mapping
                try:
                    return obj[key]  # type: ignore[index]
                except Exception:
                    pass
                return getattr(obj, key, None)

            arr = _get_field(audio_obj, "array")
            if arr is not None:
                wav = torch.as_tensor(arr).to(torch.float32)
                sr_field = _get_field(audio_obj, "sampling_rate")
                sr0 = int(sr_field) if sr_field is not None else int(self.target_sr)
            else:
                b = _get_field(audio_obj, "bytes")
                if b is not None:
                    import io
                    import soundfile as sf

                    with io.BytesIO(b) as bio:
                        wav_np, sr0 = sf.read(bio, dtype="float32", always_2d=False)
                    wav = torch.from_numpy(wav_np).to(torch.float32)
                    sr0 = int(sr0)
                else:
                    return None

            # Mono
            if wav.ndim == 2:
                if wav.shape[0] < wav.shape[1]:
                    wav = wav.mean(dim=0)
                else:
                    wav = wav.mean(dim=1)
            return wav.contiguous(), int(sr0)

        t_iter0 = time.monotonic()
        ds = self._load_hf_iterable()

        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)

        rank, world = _dist_rank_world()
        ds = _shard_hf_iterable(ds, rank, world, worker_id, num_workers)

        # IMPORTANT: shard *before* shuffle (same as `hoyeol/`).
        if self.shuffle and hasattr(ds, "shuffle"):
            ds = ds.shuffle(buffer_size=int(self.shuffle_buffer_size), seed=int(self.seed))

        rng = random.Random(int(self.seed) + rank * 10_000 + worker_id)

        log_this = (rank == 0 and worker_id == 0)
        if log_this:
            logger.warning(
                "[MLS streaming] iter start (rank=%d/%d worker=%d/%d) shuffle=%s buffer_size=%d",
                rank,
                world,
                worker_id,
                num_workers,
                bool(self.shuffle),
                int(self.shuffle_buffer_size),
            )
            logger.warning("[MLS streaming] fetching first sample (may block during HF shuffle warmup)...")

        first_yielded = False
        skipped = 0
        max_skips_before_error = 1000  # hard-stop to avoid infinite "skipping forever"
        for ex in ds:
            # Decode audio in either decoded-waveform or bytes form.
            audio = ex.get("audio") if hasattr(ex, "get") else None
            try:
                decoded = _decode_audio_to_wav_sr(audio)
            except Exception:
                decoded = None
            if decoded is None:
                skipped += 1
                if log_this and skipped in (1, 2, 3, 10, 100):
                    try:
                        ex_keys = list(ex.keys()) if hasattr(ex, "keys") else []
                    except Exception:
                        ex_keys = []
                    try:
                        a_type = type(audio).__name__
                    except Exception:
                        a_type = "<?>"
                    try:
                        a_repr = repr(audio)
                        if len(a_repr) > 200:
                            a_repr = a_repr[:200] + "..."
                    except Exception:
                        a_repr = "<repr failed>"
                    logger.warning(
                        "[MLS streaming] skip example (skipped=%d). ex_keys=%s audio_type=%s audio_repr=%s",
                        skipped,
                        ex_keys,
                        a_type,
                        a_repr,
                    )
                if skipped >= max_skips_before_error:
                    raise RuntimeError(
                        f"MLS streaming: skipped {skipped} examples without decoding audio. "
                        "This likely indicates an unexpected HF audio object format."
                    )
                continue
            wav, sr = decoded

            fid = ex.get("id", None)
            if fid is None:
                fid = ex.get("file", None)
            if fid is None:
                fid = ex.get("original_path", "")
            fid = str(fid)

            t0 = time.monotonic()
            if sr != int(self.target_sr):
                wav = torchaudio.functional.resample(wav, sr, int(self.target_sr))
                wav = wav.clamp(-1.0, 1.0)

            length = int(wav.shape[0])
            if length < int(self.min_audio_length):
                wav = F.pad(wav, (0, int(self.min_audio_length) - length))
                length = int(wav.shape[0])

            # Train-time crop matching the baseline batching behavior.
            start = 0 if length == int(self.min_audio_length) else rng.randint(0, length - int(self.min_audio_length))
            target_wav = wav[start : start + int(self.min_audio_length)].clamp(-1.0, 1.0)

            if log_this and not first_yielded:
                first_yielded = True
                # Print a compact schema snapshot (what MLS examples look like).
                try:
                    bt = float(ex.get("begin_time", 0.0))
                    et = float(ex.get("end_time", 0.0))
                    dur = float(ex.get("audio_duration", 0.0))
                    spk = ex.get("speaker_id", None)
                    audio_type = type(audio).__name__ if audio is not None else "None"
                    audio_repr = repr(audio)[:200] if audio is not None else "None"
                except Exception:
                    bt, et, dur, spk, audio_type, audio_repr = 0.0, 0.0, 0.0, None, "<?>", "<repr failed>"
                logger.warning(
                    "[MLS streaming] first yield after %.2fs (preprocess=%.2fms) fid=%s begin=%.3f end=%.3f audio_duration=%.3f speaker_id=%s audio_type=%s audio_repr=%s",
                    time.monotonic() - t_iter0,
                    (time.monotonic() - t0) * 1000.0,
                    fid,
                    bt,
                    et,
                    dur,
                    str(spk),
                    audio_type,
                    audio_repr,
                )

            yield {
                "fid": fid,
                "wav": target_wav,
                # Keep schema compatible with non-streaming SDPCodec batches.
                # (the model accesses this key unconditionally).
                "wav_24k": None,
                "ref_wav": target_wav,
                "target_id": None,
            }


class LocalTrainIterableDataset(IterableDataset):
    """
    Wrap a local/random-access HF dataset in an IterableDataset so it can be
    mixed with streaming sources during train.
    """

    def __init__(
        self,
        cfg: Any,
        dataset: Any,
        *,
        shuffle: bool,
        seed: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.target_sr = _target_sr_from_cfg(cfg)
        self.min_audio_length = int(getattr(cfg.dataset, "min_audio_length", 64000))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)

        rank, world = _dist_rank_world()
        num_shards = int(world) * int(num_workers)
        shard_index = int(rank) * int(num_workers) + int(worker_id)

        indices = list(range(shard_index, len(self.dataset), num_shards))
        rng = random.Random(int(self.seed) + rank * 10_000 + worker_id)
        if self.shuffle:
            rng.shuffle(indices)

        for idx in indices:
            ex = self.dataset[int(idx)]
            audio = ex.get("audio") if hasattr(ex, "get") else None
            if audio is None:
                continue

            arr = audio.get("array") if isinstance(audio, dict) else getattr(audio, "array", None)
            if arr is None:
                continue

            wav = torch.as_tensor(arr).to(torch.float32)
            sr = audio.get("sampling_rate") if isinstance(audio, dict) else getattr(audio, "sampling_rate", None)
            sr = int(sr) if sr is not None else int(self.target_sr)

            if wav.ndim == 2:
                if wav.shape[0] < wav.shape[1]:
                    wav = wav.mean(dim=0)
                else:
                    wav = wav.mean(dim=1)

            if sr != int(self.target_sr):
                wav = torchaudio.functional.resample(wav, sr, int(self.target_sr))
                wav = wav.clamp(-1.0, 1.0)

            length = int(wav.shape[0])
            if length < int(self.min_audio_length):
                wav = F.pad(wav, (0, int(self.min_audio_length) - length))
                length = int(wav.shape[0])

            start = 0 if length == int(self.min_audio_length) else rng.randint(0, length - int(self.min_audio_length))
            target_wav = wav[start : start + int(self.min_audio_length)].clamp(-1.0, 1.0)

            fid = ex.get("id", None) if hasattr(ex, "get") else None
            if fid is None:
                fid = ex.get("file", None) if hasattr(ex, "get") else None
            if fid is None:
                fid = ex.get("original_path", "") if hasattr(ex, "get") else ""

            yield {
                "fid": str(fid),
                "wav": target_wav,
                "wav_24k": None,
                "ref_wav": target_wav,
                "target_id": None,
            }


class CombinedStreamingTrainDataset(IterableDataset):
    """
    Mix multiple iterable train sources without requiring equal lengths.
    """

    def __init__(self, datasets: List[IterableDataset], *, seed: int):
        super().__init__()
        self.datasets = list(datasets)
        self.seed = int(seed)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        rank, _ = _dist_rank_world()
        rng = random.Random(int(self.seed) + rank * 10_000 + worker_id)

        active = [iter(ds) for ds in self.datasets]
        while active:
            pick = rng.randrange(len(active))
            try:
                yield next(active[pick])
            except StopIteration:
                active.pop(pick)


def _collate_bicodec_streaming(bs: List[Dict[str, Any]]) -> Dict[str, Any]:
    import torch

    fids = [b.get("fid", "") for b in bs]
    wavs = torch.stack([b["wav"] for b in bs])
    ref_wavs = torch.stack([b["ref_wav"] for b in bs])
    wavs = wavs.clamp(-1.0, 1.0)
    ref_wavs = ref_wavs.clamp(-1.0, 1.0)
    # IMPORTANT: keep keys consistent with map-style dataloaders.
    return {
        "fid": fids,
        "wav": wavs,
        "wav_24k": None,
        "ref_wav": ref_wavs,
        "target_id": None,
        # Validation-only keys (keep present to avoid KeyError in shared code paths)
        "ref_wav_vc": None,
        "vc_target_id": None,
    }


def _resolve_hf_datasets_cache_dir(cfg) -> str:
    """
    Resolve the datasets cache directory deterministically.

    - If cfg.cache_dir is set, use it.
    - Else, follow HF env vars (HF_DATASETS_CACHE preferred, otherwise HF_HOME/datasets).
    This avoids confusing logs like "Loading dataset in None..." and ensures cache is shared
    correctly between host/container based on env configuration.
    """
    cache_dir = getattr(cfg, "cache_dir", None)
    if cache_dir:
        return str(cache_dir)
    hf_datasets_cache = os.environ.get("HF_DATASETS_CACHE")
    if hf_datasets_cache:
        return str(hf_datasets_cache)
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return str(os.path.join(hf_home, "datasets"))


def _target_sr_from_cfg(cfg: Any, default: int = 16000) -> int:
    """
    Best-effort extraction of the project's target audio sample-rate from Hydra cfg.
    Falls back to 16kHz which is what most HF ASR corpora use.
    """
    try:
        preprocess = getattr(cfg, "preprocess", None)
        audio = getattr(preprocess, "audio", None)
        sr = getattr(audio, "sr", None)
        if sr is not None:
            return int(sr)
    except Exception:
        raise ValueError(f"Failed to get sample rate from cfg.preprocess.audio.sr. cfg={cfg}")
    try:
        return int(getattr(getattr(cfg, "dataset", None), "sample_rate", default))
    except Exception:
        raise ValueError(f"Failed to get sample rate from cfg.dataset.sample_rate. cfg={cfg}")


def _keep_audio_and_id(ds, *, target_sr: Optional[int] = None):
    """
    Normalize HF ASR datasets to a minimal common schema so we can concatenate
    different datasets (e.g., LibriSpeech + MLS) safely.

    Required columns for this project:
    - audio: dict with 'array' and 'sampling_rate'
    - id: unique sample identifier (used for logging/VC pairing)
    """
    cols = list(getattr(ds, "column_names", []))
    if "id" not in cols and "file" in cols:
        ds = ds.rename_column("file", "id")
        cols = list(getattr(ds, "column_names", []))
    # Some MLS variants (e.g. parler-tts/mls_eng) use original_path instead of file/id
    if "id" not in cols and "original_path" in cols:
        ds = ds.rename_column("original_path", "id")
        cols = list(getattr(ds, "column_names", []))
    if "audio" not in cols or "id" not in cols:
        raise ValueError(
            f"Dataset must contain 'audio' and 'id' columns. Got columns={cols}"
        )
    drop_cols = [c for c in cols if c not in ("audio", "id")]
    if len(drop_cols) > 0:
        ds = ds.remove_columns(drop_cols)

    # IMPORTANT:
    # HF datasets can expose `Audio(sampling_rate=None)` depending on how the dataset builder
    # defines the feature or how it was cached. When we concatenate multiple corpora, HF requires
    # exact feature alignment, so we cast the audio column to a single target sampling-rate.
    if target_sr is not None:
        try:
            from datasets import Audio

            features = getattr(ds, "features", None)
            audio_feat = None
            if isinstance(features, dict):
                audio_feat = features.get("audio")
            # If this is an `Audio` feature with missing/other SR, cast to the expected SR.
            if isinstance(audio_feat, Audio):
                sr0 = getattr(audio_feat, "sampling_rate", None)
                if sr0 is None or int(sr0) != int(target_sr):
                    ds = ds.cast_column("audio", Audio(sampling_rate=int(target_sr), decode=True))
        except Exception:
            logger.warning(
                "Failed to cast HF dataset `audio` column to target_sr=%s; continuing without cast.",
                str(target_sr),
                exc_info=True,
            )
    return ds


def _keep_audio_id_and_speaker(ds, *, target_sr: Optional[int] = None):
    """
    Like `_keep_audio_and_id`, but also preserves `speaker_id`.

    Used for datasets where speaker_id is the canonical speaker field (e.g. VCTK).
    """
    cols = list(getattr(ds, "column_names", []))
    if "id" not in cols and "file" in cols:
        ds = ds.rename_column("file", "id")
        cols = list(getattr(ds, "column_names", []))
    if "id" not in cols and "original_path" in cols:
        ds = ds.rename_column("original_path", "id")
        cols = list(getattr(ds, "column_names", []))
    if "audio" not in cols or "id" not in cols or "speaker_id" not in cols:
        raise ValueError(
            f"Dataset must contain 'audio', 'id', and 'speaker_id'. Got columns={cols}"
        )
    drop_cols = [c for c in cols if c not in ("audio", "id", "speaker_id")]
    if len(drop_cols) > 0:
        ds = ds.remove_columns(drop_cols)

    if target_sr is not None:
        try:
            from datasets import Audio

            features = getattr(ds, "features", None)
            audio_feat = None
            if isinstance(features, dict):
                audio_feat = features.get("audio")
            if isinstance(audio_feat, Audio):
                sr0 = getattr(audio_feat, "sampling_rate", None)
                if sr0 is None or int(sr0) != int(target_sr):
                    ds = ds.cast_column("audio", Audio(sampling_rate=int(target_sr), decode=True))
        except Exception:
            logger.warning(
                "Failed to cast HF dataset `audio` column to target_sr=%s; continuing without cast.",
                str(target_sr),
                exc_info=True,
            )
    return ds


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(f"[timeit] {func.__qualname__} took {elapsed_ms:.2f} ms")
    return wrapper


def _extract_speaker_ids(ds) -> List[str]:
    """
    speaker_id 컬럼이 있으면 그대로 사용.
    없으면 id에서 첫 토큰(첫 '-' 또는 '_' 전)을 speaker로 파싱.
    LibriSpeech: '7789-62046-0000' -> '7789'
    LibriTTS:    '121_000000'      -> '121'
    """
    cols = getattr(ds, "column_names", [])
    if 'speaker_id' in cols:
        return [str(s) for s in ds['speaker_id']]

    ids = ds['id']
    spk_ids: List[str] = []
    for i in ids:
        s = str(i)
        # 숫자-id가 대부분이므로 우선 숫자+구분자 패턴 매칭
        m = re.match(r'^(\d+)[-_]', s)
        if m:
            spk_ids.append(m.group(1))
            continue
        # 일반 fallback: 첫 '-' 또는 '_' 전까지
        spk_ids.append(re.split(r'[-_]', s)[0])
    return spk_ids


def add_vc_pair_in_test_dataset(cfg, test_dataset, enforce_min_ref_filter: bool = False):
    # 1) speaker_id 추출 (컬럼 우선, 없으면 id 파싱)
    id_list: List[str] = test_dataset['id']
    spk_ids: List[str] = _extract_speaker_ids(test_dataset)

    # 2) speaker -> indices 맵 구성
    spk2idxs = {}
    for idx, spk in enumerate(spk_ids):
        spk2idxs.setdefault(spk, []).append(idx)
    all_spks = list(spk2idxs.keys())

    # 3) 재현성 있는 랜덤 시드
    seed = getattr(cfg.dataset, "pair_seed", 42)
    rng = random.Random(seed)

    # 3.5) 최소 참조 길이(초) - 테스트에서만 적용
    min_ref_seconds = float(getattr(cfg.dataset, "min_ref_seconds", 3.0))

    # duration cache to avoid repeated decoding
    _dur_cache = {}

    def _get_duration_sec(j: int) -> float:
        if j in _dur_cache:
            return _dur_cache[j]
        sample = test_dataset[int(j)]
        arr = sample['audio']['array']
        sr = int(sample['audio']['sampling_rate'])
        dur = float(len(arr)) / float(sr)
        _dur_cache[j] = dur
        return dur

    # 4) 각 샘플마다 "다른 speaker"에서 랜덤 인덱스 선택
    pair_idx_list: List[int] = []
    for i, spk in enumerate(spk_ids):
        other_spks = [s for s in all_spks if s != spk]
        if len(other_spks) == 0:
            # 데이터셋에 화자가 1명뿐인 극단 케이스
            j = i
        else:
            if not enforce_min_ref_filter:
                # 필터 미적용: 원래대로 랜덤 선택
                other_spk = rng.choice(other_spks)
                j = rng.choice(spk2idxs[other_spk])
            else:
                j = None
                # 여러 번 시도하여 3초 이상인 샘플을 우선적으로 선택
                for _ in range(50):
                    other_spk = rng.choice(other_spks)
                    cand = rng.choice(spk2idxs[other_spk])
                    if _get_duration_sec(cand) >= min_ref_seconds:
                        j = cand
                        break
                if j is None:
                    # 실패 시, 후보군 전체에서 3초 이상 필터링 후 선택
                    eligible = []
                    for s in other_spks:
                        for cand in spk2idxs[s]:
                            if _get_duration_sec(cand) >= min_ref_seconds:
                                eligible.append(cand)
                    if len(eligible) > 0:
                        j = rng.choice(eligible)
                    else:
                        # 그래도 없으면 에러
                        raise ValueError(
                            f"No valid VC reference found for {spk_ids[i]}")
                        # 원복하려면 아래 두 줄 사용:
                        # other_spk = rng.choice(other_spks)
                        # j = rng.choice(spk2idxs[other_spk])
        pair_idx_list.append(j)

    # 5) 보조적으로 pair_id도 함께 추가
    pair_id_list: List[str] = [id_list[j] for j in pair_idx_list]

    # 6) 컬럼 추가
    test_dataset = test_dataset.add_column('vc_pair_idx', pair_idx_list)
    test_dataset = test_dataset.add_column('vc_pair_id', pair_id_list)
    return test_dataset


class VctkVCEvalDataset(Dataset):
    """
    Map-style VC evaluation dataset for VCTK:
    - Iteration yields only selected *source* utterances (possibly repeated)
    - Each yielded item contains `vc_pair_idx` that points to a base-dataset index
    - Reference utterances are fetched via `get_base_item(base_idx)`

    This supports "K refs per source utterance" without polluting the iteration
    with reference-only rows.
    """

    def __init__(self, base_dataset, source_base_indices: List[int], ref_base_indices: List[int]):
        if len(source_base_indices) != len(ref_base_indices):
            raise ValueError(
                f"source_base_indices and ref_base_indices must have same length. "
                f"Got {len(source_base_indices)} vs {len(ref_base_indices)}"
            )
        self.base_dataset = base_dataset
        self.source_base_indices = [int(x) for x in source_base_indices]
        self.ref_base_indices = [int(x) for x in ref_base_indices]

    def __len__(self):
        return len(self.source_base_indices)

    def get_base_item(self, base_idx: int) -> Dict[str, Any]:
        return self.base_dataset[int(base_idx)]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        src_base_idx = int(self.source_base_indices[int(idx)])
        ref_base_idx = int(self.ref_base_indices[int(idx)])
        ex = dict(self.base_dataset[int(src_base_idx)])
        ex["vc_pair_idx"] = ref_base_idx
        return ex


def _sample_vctk_source_indices(
    ds,
    *,
    seed: int,
    utts_per_speaker: int,
    max_speakers: Optional[int] = None,
) -> List[int]:
    """Sample utts_per_speaker random indices per speaker. Returns flat list of base-dataset indices."""
    speaker_ids = [str(s) for s in ds["speaker_id"]]
    spk2idxs: Dict[str, List[int]] = {}
    for i, spk in enumerate(speaker_ids):
        spk2idxs.setdefault(spk, []).append(int(i))

    all_spks = list(spk2idxs.keys())
    rng = random.Random(int(seed))
    rng.shuffle(all_spks)
    if max_speakers is not None:
        all_spks = all_spks[: int(max_speakers)]

    src_indices: List[int] = []
    for spk in all_spks:
        idxs = spk2idxs[spk]
        if len(idxs) == 0:
            continue
        if len(idxs) >= int(utts_per_speaker):
            src_indices.extend(rng.sample(idxs, int(utts_per_speaker)))
        else:
            # Rare edge case: speaker has too few utterances; sample with replacement.
            src_indices.extend([rng.choice(idxs) for _ in range(int(utts_per_speaker))])
    return src_indices


def _build_vctk_vc_pairs(
    ds,
    *,
    seed: int,
    source_base_indices: List[int],
    refs_per_source: int,
    enforce_min_ref_filter: bool,
    min_ref_seconds: float,
) -> tuple[List[int], List[int]]:
    """Build (source_idx, ref_idx) pairs for VCTK VC eval.

    For each source utterance, picks `refs_per_source` refs from *different* speakers.
    When enforce_min_ref_filter, refs must have duration >= min_ref_seconds.
    """
    print("[VCTK] _build_vctk_vc_pairs: start")
    speaker_ids = [str(s) for s in ds["speaker_id"]]
    spk2idxs: Dict[str, List[int]] = {}
    for i, spk in enumerate(speaker_ids):
        spk2idxs.setdefault(spk, []).append(int(i))
    all_spks = list(spk2idxs.keys())
    rng = random.Random(int(seed))

    # Duration cache: decode once per index
    dur_cache: Dict[int, float] = {}

    def _duration_sec(idx: int) -> float:
        if idx in dur_cache:
            return dur_cache[idx]
        sample = ds[int(idx)]
        arr = sample["audio"]["array"]
        sr = int(sample["audio"]["sampling_rate"])
        dur_cache[idx] = float(len(arr)) / float(sr)
        return dur_cache[idx]

    need_dur_check = bool(enforce_min_ref_filter) and float(min_ref_seconds) > 0.0

    def _eligible_indices(src_spk: str, exclude: set[int]) -> List[int]:
        """Indices from other speakers, optionally filtered by min_ref_seconds."""
        others = [s for s in all_spks if s != src_spk]
        if not others:
            return []
        out: List[int] = []
        for spk in others:
            for idx in spk2idxs[spk]:
                if idx in exclude:
                    continue
                if need_dur_check and _duration_sec(idx) < float(min_ref_seconds):
                    continue
                out.append(int(idx))
        return out

    src_rep: List[int] = []
    ref_list: List[int] = []
    n_src = len(source_base_indices)
    for ii, src_base_idx in enumerate(source_base_indices):
        if ii % 100 == 0 or ii == n_src - 1:
            print(f"[VCTK] _build_vctk_vc_pairs: {ii+1}/{n_src} sources processed")
        src_base_idx = int(src_base_idx)
        src_spk = str(speaker_ids[src_base_idx])
        used_refs: set[int] = set()

        for _ in range(int(refs_per_source)):
            eligible = _eligible_indices(src_spk, used_refs)
            if not eligible:
                msg = f"VCTK VC pairing failed: no eligible ref for src_spk={src_spk}"
                if enforce_min_ref_filter:
                    msg += f" (need duration >= {min_ref_seconds:.2f}s)"
                raise ValueError(msg)
            picked = rng.choice(eligible)
            used_refs.add(picked)
            src_rep.append(src_base_idx)
            ref_list.append(picked)

    print("[VCTK] _build_vctk_vc_pairs: done")
    return src_rep, ref_list


def load_librispeech_dataset(cfg, dc):
    cache_dir = _resolve_hf_datasets_cache_dir(cfg)
    print(f"Loading dataset in {cache_dir}...")

    # Backward-compatible helper: historically returned (train, val, test).
    # Prefer using `load_librispeech_splits(...)` to selectively load splits.
    return load_librispeech_splits(cfg, dc, load_train=True, load_val=True, load_test=True)


def load_librispeech_splits(cfg, dc, *, load_train: bool, load_val: bool, load_test: bool):
    cache_dir = _resolve_hf_datasets_cache_dir(cfg)
    print(f"Loading dataset in {cache_dir}...")

    train_dataset = None
    val_dataset = None
    test_dataset = None

    if load_train:
        train_clean_100 = _load_dataset_with_token('openslr/librispeech_asr',
                                       split="train.clean.100",
                                       num_proc=cfg.dataset.train.num_workers,
                                       cache_dir=cache_dir,
                                       download_config=dc)
        train_clean_360 = _load_dataset_with_token('openslr/librispeech_asr',
                                       split="train.clean.360",
                                       num_proc=cfg.dataset.train.num_workers,
                                       cache_dir=cache_dir,
                                       download_config=dc)
        train_other_500 = _load_dataset_with_token('openslr/librispeech_asr',
                                       split="train.other.500",
                                       num_proc=cfg.dataset.train.num_workers,
                                       cache_dir=cache_dir,
                                       download_config=dc)
        train_dataset = concatenate_datasets(
            [train_clean_100, train_clean_360, train_other_500])

    if load_val:
        val_dataset = _load_dataset_with_token('openslr/librispeech_asr',
                                   split="validation.clean",
                                   num_proc=cfg.dataset.val.num_workers,
                                   cache_dir=cache_dir,
                                   download_config=dc)

    if load_test:
        test_dataset = _load_dataset_with_token('openslr/librispeech_asr',
                                    split="test.clean",
                                    num_proc=cfg.dataset.test.num_workers,
                                    cache_dir=cache_dir,
                                    download_config=dc)

    return train_dataset, val_dataset, test_dataset


def load_libritts_dataset(cfg, dc):
    cache_dir = _resolve_hf_datasets_cache_dir(cfg)
    print(f"Loading dataset in {cache_dir}...")

    # Backward-compatible helper: historically returned (train, val, test).
    # Prefer using `load_libritts_splits(...)` to selectively load splits.
    return load_libritts_splits(cfg, dc, load_train=True, load_val=True, load_test=True)


def load_libritts_splits(cfg, dc, *, load_train: bool, load_val: bool, load_test: bool):
    cache_dir = _resolve_hf_datasets_cache_dir(cfg)
    print(f"Loading dataset in {cache_dir}...")

    train_dataset = None
    val_dataset = None
    test_dataset = None

    if load_train:
        train_clean_100 = _load_dataset_with_token('mythicinfinity/libritts', 'clean', split="train.clean.100",
                                       num_proc=cfg.dataset.train.num_workers, cache_dir=cache_dir, download_config=dc)
        train_clean_360 = _load_dataset_with_token('mythicinfinity/libritts', 'clean', split="train.clean.360",
                                       num_proc=cfg.dataset.train.num_workers, cache_dir=cache_dir, download_config=dc)
        train_other_500 = _load_dataset_with_token('mythicinfinity/libritts', 'other', split="train.other.500",
                                       num_proc=cfg.dataset.train.num_workers, cache_dir=cache_dir, download_config=dc)
        train_dataset = concatenate_datasets(
            [train_clean_100, train_clean_360, train_other_500])

    if load_val:
        val_dataset = _load_dataset_with_token('mythicinfinity/libritts', 'dev', split="dev.clean",
                                   num_proc=cfg.dataset.val.num_workers, cache_dir=cache_dir, download_config=dc)

    if load_test:
        test_dataset = _load_dataset_with_token('mythicinfinity/libritts', 'clean', split="test.clean",
                                    num_proc=cfg.dataset.test.num_workers, cache_dir=cache_dir, download_config=dc)

    return train_dataset, val_dataset, test_dataset


def load_mls_splits(cfg, dc, *, load_train: bool, load_val: bool, load_test: bool):
    """
    Load MLS (MultiLingual LibriSpeech) splits from HuggingFace.

    Config expected under `cfg.dataset.mls` (all optional):
    - dataset_id: str. Default: "facebook/multilingual_librispeech"
        - "facebook/multilingual_librispeech": multilingual (configs are languages like german/french/...)
        - "parler-tts/mls_eng": English-only MLS (no language config; only split)
    - languages: list[str] or a single str. Default: ["english"]
    - train_split: str. Default: "train"
    - val_split: str. Default: "dev"
    - test_split: str. Default: "test"
    - num_proc_train/num_proc_val/num_proc_test: optional HF dataset build workers.
      Defaults to the corresponding DataLoader worker count.
    """
    cache_dir = _resolve_hf_datasets_cache_dir(cfg)
    print(f"Loading dataset in {cache_dir}...")
    target_sr = _target_sr_from_cfg(cfg)

    mls_cfg = cfg.dataset.get("mls", {})
    dataset_id = str(mls_cfg.get("dataset_id", "facebook/multilingual_librispeech"))
    languages = mls_cfg.get("languages", ["english"])
    if isinstance(languages, str):
        languages = [languages]
    languages = [str(x).lower() for x in languages]
    # languages are only used for multilingual configs
    if dataset_id == "facebook/multilingual_librispeech" and len(languages) == 0:
        raise ValueError("cfg.dataset.mls.languages must be non-empty for facebook/multilingual_librispeech")

    train_split = str(mls_cfg.get("train_split", "train"))
    val_split = str(mls_cfg.get("val_split", "dev"))
    test_split = str(mls_cfg.get("test_split", "test"))
    train_num_proc = int(mls_cfg.get("num_proc_train", cfg.dataset.train.num_workers))
    val_num_proc = int(mls_cfg.get("num_proc_val", cfg.dataset.val.num_workers))
    test_num_proc = int(mls_cfg.get("num_proc_test", cfg.dataset.test.num_workers))

    def _load_one(split: str, *, num_proc: int):
        # English-only MLS (parler-tts/mls_eng) has no language config.
        if dataset_id == "parler-tts/mls_eng":
            ds = _load_dataset_with_token(
                dataset_id,
                split=split,
                num_proc=num_proc,
                cache_dir=cache_dir,
                download_config=dc,
            )
            return _keep_audio_and_id(ds, target_sr=target_sr)

        # Default multilingual MLS.
        dsets = []
        for lang in languages:
            ds = _load_dataset_with_token(
                dataset_id,
                lang,
                split=split,
                num_proc=num_proc,
                cache_dir=cache_dir,
                download_config=dc,
            )
            ds = _keep_audio_and_id(ds, target_sr=target_sr)
            dsets.append(ds)
        if len(dsets) == 1:
            return dsets[0]
        return concatenate_datasets(dsets)

    train_dataset = None
    val_dataset = None
    test_dataset = None

    if load_train:
        train_dataset = _load_one(train_split, num_proc=train_num_proc)
    if load_val:
        val_dataset = _load_one(val_split, num_proc=val_num_proc)
    if load_test:
        test_dataset = _load_one(test_split, num_proc=test_num_proc)
    return train_dataset, val_dataset, test_dataset


def load_vctk_splits(cfg, dc, *, load_train: bool, load_val: bool, load_test: bool):
    """Load VCTK from HuggingFace (sanchit-gandhi/vctk). mic1 only; id = p225_001_mic1. Eval only.
    VCTK uses project .cache (execution folder) instead of HF_HOME/~/.cache."""
    # Project root: ptl/bicodec/data_module.py -> repo root
    repo_root = Path(__file__).resolve().parent.parent.parent
    cache_dir = str(repo_root / ".cache" / "huggingface" / "datasets")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Loading VCTK dataset in {cache_dir}...")
    target_sr = _target_sr_from_cfg(cfg)

    vctk_cfg = cfg.dataset.get("vctk", {})
    dataset_id = str(vctk_cfg.get("dataset_id", "sanchit-gandhi/vctk"))
    split = str(vctk_cfg.get("split", "train"))

    train_dataset = None
    val_dataset = None
    test_dataset = None

    def _load_one(*, num_proc: int):
        ds = _load_dataset_with_token(
            dataset_id,
            split=split,
            num_proc=num_proc,
            cache_dir=cache_dir,
            download_config=dc,
        )
        ds = _keep_audio_id_and_speaker(ds, target_sr=target_sr)
        # CRITICAL: we only use mic1. mic2 is filtered out and never used for pairing.
        print("[VCTK] load_vctk_splits: filter(mic1)...")
        ds = ds.filter(_vctk_keep_mic1, num_proc=max(1, int(num_proc)))
        # Make ids compact: use basename without extension (e.g. p225_001_mic1).
        print("[VCTK] load_vctk_splits: map(shorten_id)...")
        ds = ds.map(_vctk_shorten_id, num_proc=max(1, int(num_proc)))
        print("[VCTK] load_vctk_splits: done")
        return ds

    # VCTK is eval-only here: we do not support training on it via this DataModule.
    if load_train:
        raise ValueError("dataset.name='vctk' is intended for evaluation only (val/test).")
    if load_val:
        val_dataset = _load_one(num_proc=int(cfg.dataset.val.num_workers))
    if load_test:
        test_dataset = _load_one(num_proc=int(cfg.dataset.test.num_workers))
    return train_dataset, val_dataset, test_dataset


def _optimizer_steps_from_checkpoint(ckpt: dict) -> int:
    """Optimizer update count for resume (SdpCodec manual opt uses total_step, not global_step)."""
    state = ckpt.get("state_dict") or {}
    for key, value in state.items():
        if key == "total_step" or key.endswith(".total_step"):
            return int(value.item() if torch.is_tensor(value) else value)
    return int(ckpt.get("global_step", 0))


def _manual_gradient_accumulation_steps(cfg) -> int:
    """Manual-optimization modules use cfg.train.gradient_accumulation_steps."""
    try:
        return max(1, int(getattr(cfg.train, "gradient_accumulation_steps", 1)))
    except Exception:
        return 1


def _sampler_batches_per_epoch(
    *,
    dataset_len: int,
    batch_size: int,
    shuffle: bool,
    num_replicas: int,
) -> int:
    n = int(dataset_len)
    batch_size = max(1, int(batch_size))
    num_replicas = max(1, int(num_replicas))
    if shuffle and num_replicas > 1:
        # Matches ResumableDistributedSampler(drop_last=True) + DataLoader(drop_last=True).
        per_rank_samples = (n // num_replicas)
    else:
        per_rank_samples = n
    return max(1, per_rank_samples // batch_size)


def _sampler_state_from_optimizer_steps(
    *,
    opt_steps: int,
    accum: int,
    batch_size: int,
    dataset_len: int,
    shuffle: bool,
    num_replicas: int,
) -> Dict[str, int]:
    batches_per_epoch = _sampler_batches_per_epoch(
        dataset_len=dataset_len,
        batch_size=batch_size,
        shuffle=shuffle,
        num_replicas=num_replicas,
    )
    batches_consumed = int(opt_steps) * max(1, int(accum))
    step_in_epoch = int(batches_consumed % batches_per_epoch)
    return {
        "epoch": int(batches_consumed // batches_per_epoch),
        "cursor": int(step_in_epoch * max(1, int(batch_size))),
        "batches_per_epoch": int(batches_per_epoch),
        "batches_consumed": int(batches_consumed),
    }


def _verify_distributed_sampler_shards(sampler: Sampler, *, preview: int = 64) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    world = int(dist.get_world_size())
    if world <= 1 or not isinstance(sampler, ResumableDistributedSampler):
        return

    rank = int(dist.get_rank())
    start = int(getattr(sampler, "_cursor", 0))
    rank_indices = list(getattr(sampler, "_rank_indices", []))
    local_preview = [int(x) for x in rank_indices[start:start + int(preview)]]
    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, local_preview)

    failed = torch.zeros((), dtype=torch.int64, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    error_msg = ""
    if rank == 0:
        flat = []
        for shard in gathered:
            flat.extend(shard or [])
        duplicate_count = len(flat) - len(set(flat))
        lengths = [len(shard or []) for shard in gathered]
        if duplicate_count:
            failed.fill_(1)
            error_msg = (
                "ResumableDistributedSampler shard overlap detected in preview "
                f"(world={world}, preview={preview}, duplicates={duplicate_count}, lengths={lengths})."
            )
        else:
            print(
                colored(
                    f"[sampler] DDP shard preview OK: world={world}, preview={preview}, "
                    f"per_rank_preview_lengths={lengths}",
                    "cyan",
                )
            )
    dist.broadcast(failed, src=0)
    if int(failed.item()) != 0:
        raise RuntimeError(
            error_msg
            or "ResumableDistributedSampler shard overlap detected in preview on rank 0."
        )


def _restore_train_sampler_from_ckpt(
    sampler: Sampler,
    ckpt: dict,
    *,
    batch_size: int,
    dataset: Dataset,
    phase_cfg,
    accum: int,
) -> None:
    """Restore mid-epoch sampler position on resume."""
    opt_steps = _optimizer_steps_from_checkpoint(ckpt)
    num_replicas = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    if opt_steps > 0:
        state = _sampler_state_from_optimizer_steps(
            opt_steps=opt_steps,
            accum=accum,
            batch_size=batch_size,
            dataset_len=len(dataset),
            shuffle=bool(phase_cfg.shuffle),
            num_replicas=num_replicas,
        )
        sampler.load_state_dict(state)
        global_step = int(ckpt.get("global_step", 0))
        logger.info(
            "Restored sampler from optimizer steps "
            "(total_step=%d, global_step=%d, accum=%d, batches_per_epoch=%d, "
            "batches_consumed=%d -> epoch=%d, cursor=%d).",
            opt_steps,
            global_step,
            accum,
            state["batches_per_epoch"],
            state["batches_consumed"],
            state["epoch"],
            state["cursor"],
        )
        return

    saved = ckpt.get("datamodule_sampler_state")
    if saved is not None:
        sampler.load_state_dict(saved)
        logger.info(
            "Restored sampler from datamodule_sampler_state fallback (epoch=%d, cursor=%d).",
            int(saved.get("epoch", 0)),
            int(saved.get("cursor", 0)),
        )


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        ocwd = _original_cwd()
        self.ocwd = ocwd
        offline_env = (os.environ.get("HF_HUB_OFFLINE", "0") == "1" or
                       os.environ.get("HF_DATASETS_OFFLINE", "0") == "1")
        dc = DownloadConfig(
            max_retries=20,          # 네트워크 흔들림 대비 재시도 늘리기
            resume_download=True,    # 이어받기 강제
            use_etag=True,           # 캐시 무결성/재개용
            local_files_only=offline_env,  # 오프라인 모드 시 캐시만 사용
        )
        self.dc = dc

        raw_name = str(self.cfg.dataset.get("name", "librispeech")).lower()
        # Normalize aliases (historical)
        if raw_name in ("mls+librispeech", "librispeech+mls", "all(mls+librispeech)"):
            raw_name = "all"
        self.dataset_name = raw_name

        if self.dataset_name == 'librispeech':
            print(colored("Using librispeech dataset.",
                          "green", attrs=['bold']))
        elif self.dataset_name == 'libritts':
            print(colored("Using libritts dataset.",
                          "green", attrs=['bold']))
        elif self.dataset_name == 'mls':
            print(colored("Using MLS dataset.",
                          "green", attrs=['bold']))
        elif self.dataset_name == 'all':
            print(colored("Using combined dataset: LibriSpeech + MLS (train only).",
                          "green", attrs=['bold']))
        elif self.dataset_name == 'all_streaming':
            print(colored("Using combined streaming dataset: LibriSpeech + MLS (train only).",
                          "green", attrs=['bold']))
        elif self.dataset_name == 'vctk':
            print(colored("Using VCTK dataset (evaluation only).",
                          "green", attrs=['bold']))
        else:
            raise ValueError(
                f"Unsupported dataset: {self.dataset_name}. Supported datasets are librispeech, libritts, mls, all, all_streaming, vctk."
            )

        # Lazy-loaded by `setup(stage)` (prevents downloading train splits during `trainer.test()`).
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._train_sampler = None  # ResumableSampler ref for checkpoint save (train only)

    def setup(self, stage: Optional[str] = None):
        """
        Load datasets lazily and only what is needed for the current stage.
        This prevents `trainer.test()` from downloading/loading train splits.
        """
        # Normalize stage string
        if stage is not None:
            stage = str(stage).lower()

        def _load_splits(*, load_train: bool, load_val: bool, load_test: bool):
            target_sr = _target_sr_from_cfg(self.cfg)
            if self.dataset_name == "librispeech":
                return load_librispeech_splits(
                    self.cfg,
                    self.dc,
                    load_train=load_train,
                    load_val=load_val,
                    load_test=load_test,
                )
            if self.dataset_name == "libritts":
                return load_libritts_splits(
                    self.cfg,
                    self.dc,
                    load_train=load_train,
                    load_val=load_val,
                    load_test=load_test,
                )
            if self.dataset_name == "vctk":
                return load_vctk_splits(
                    self.cfg,
                    self.dc,
                    load_train=load_train,
                    load_val=load_val,
                    load_test=load_test,
                )
            if self.dataset_name == "mls":
                # Train on MLS, but always validate/test on LibriSpeech.
                # (Keeps evaluation stable and avoids schema/metadata mismatch concerns.)
                train_dataset = None
                val_dataset = None
                test_dataset = None

                if load_train:
                    mls_cfg = self.cfg.dataset.get("mls", {})
                    streaming = bool(mls_cfg.get("streaming", False)) if hasattr(mls_cfg, "get") else bool(getattr(mls_cfg, "streaming", False))
                    if streaming:
                        dataset_id = str(mls_cfg.get("dataset_id", "facebook/multilingual_librispeech"))
                        languages = mls_cfg.get("languages", ["english"])
                        if isinstance(languages, str):
                            languages = [languages]
                        train_split = str(mls_cfg.get("train_split", "train"))
                        seed = int(mls_cfg.get("seed", 1024))
                        shuffle_buffer_size = int(mls_cfg.get("shuffle_buffer_size", 2_000))
                        train_dataset = MlsStreamingTrainDataset(
                            self.cfg,
                            self.dc,
                            dataset_id=dataset_id,
                            languages=[str(x).lower() for x in languages],
                            split=train_split,
                            shuffle=bool(getattr(self.cfg.dataset.train, "shuffle", True)),
                            shuffle_buffer_size=shuffle_buffer_size,
                            seed=seed,
                        )
                    else:
                        train_dataset, _, _ = load_mls_splits(
                            self.cfg, self.dc, load_train=True, load_val=False, load_test=False
                        )
                        train_dataset = _keep_audio_and_id(train_dataset, target_sr=target_sr)

                if load_val:
                    _, val_ls, _ = load_librispeech_splits(
                        self.cfg, self.dc, load_train=False, load_val=True, load_test=False
                    )
                    val_dataset = _keep_audio_and_id(val_ls, target_sr=target_sr)

                if load_test:
                    _, _, test_ls = load_librispeech_splits(
                        self.cfg, self.dc, load_train=False, load_val=False, load_test=True
                    )
                    test_dataset = _keep_audio_and_id(test_ls, target_sr=target_sr)

                return train_dataset, val_dataset, test_dataset

            if self.dataset_name == "all":
                # Train: LibriSpeech + MLS
                # Val/Test: always LibriSpeech (stable eval; avoid schema mismatch)
                train_dataset = None
                val_dataset = None
                test_dataset = None

                if load_train:
                    mls_cfg = self.cfg.dataset.get("mls", {})
                    streaming = bool(mls_cfg.get("streaming", False)) if hasattr(mls_cfg, "get") else bool(getattr(mls_cfg, "streaming", False))
                    if streaming:
                        raise ValueError(
                            "cfg.dataset.name='all' with cfg.dataset.mls.streaming=True is not supported. "
                            "To combine datasets, both sources must support the same (streaming vs map-style) interface."
                        )
                    train_ls, _, _ = load_librispeech_splits(
                        self.cfg, self.dc, load_train=True, load_val=False, load_test=False
                    )
                    train_mls, _, _ = load_mls_splits(
                        self.cfg, self.dc, load_train=True, load_val=False, load_test=False
                    )
                    train_dataset = concatenate_datasets(
                        [
                            _keep_audio_and_id(train_ls, target_sr=target_sr),
                            _keep_audio_and_id(train_mls, target_sr=target_sr),
                        ]
                    )

                if load_val:
                    _, val_ls, _ = load_librispeech_splits(
                        self.cfg, self.dc, load_train=False, load_val=True, load_test=False
                    )
                    val_dataset = _keep_audio_and_id(val_ls, target_sr=target_sr)

                if load_test:
                    _, _, test_ls = load_librispeech_splits(
                        self.cfg, self.dc, load_train=False, load_val=False, load_test=True
                    )
                    test_dataset = _keep_audio_and_id(test_ls, target_sr=target_sr)

                return train_dataset, val_dataset, test_dataset

            if self.dataset_name == "all_streaming":
                # Train: LibriSpeech(iterable from local HF cache) + MLS(streaming)
                # Val/Test: always LibriSpeech (stable eval; avoid schema mismatch)
                train_dataset = None
                val_dataset = None
                test_dataset = None

                if load_train:
                    mls_cfg = self.cfg.dataset.get("mls", {})
                    dataset_id = str(mls_cfg.get("dataset_id", "facebook/multilingual_librispeech"))
                    languages = mls_cfg.get("languages", ["english"])
                    if isinstance(languages, str):
                        languages = [languages]
                    train_split = str(mls_cfg.get("train_split", "train"))
                    seed = int(mls_cfg.get("seed", 1024))
                    shuffle_buffer_size = int(mls_cfg.get("shuffle_buffer_size", 2_000))

                    train_ls, _, _ = load_librispeech_splits(
                        self.cfg, self.dc, load_train=True, load_val=False, load_test=False
                    )
                    train_ls = _keep_audio_and_id(train_ls, target_sr=target_sr)

                    ls_iterable = LocalTrainIterableDataset(
                        self.cfg,
                        train_ls,
                        shuffle=bool(getattr(self.cfg.dataset.train, "shuffle", True)),
                        seed=seed,
                    )
                    mls_iterable = MlsStreamingTrainDataset(
                        self.cfg,
                        self.dc,
                        dataset_id=dataset_id,
                        languages=[str(x).lower() for x in languages],
                        split=train_split,
                        shuffle=bool(getattr(self.cfg.dataset.train, "shuffle", True)),
                        shuffle_buffer_size=shuffle_buffer_size,
                        seed=seed,
                    )
                    train_dataset = CombinedStreamingTrainDataset(
                        [ls_iterable, mls_iterable],
                        seed=seed,
                    )

                if load_val:
                    _, val_ls, _ = load_librispeech_splits(
                        self.cfg, self.dc, load_train=False, load_val=True, load_test=False
                    )
                    val_dataset = _keep_audio_and_id(val_ls, target_sr=target_sr)

                if load_test:
                    _, _, test_ls = load_librispeech_splits(
                        self.cfg, self.dc, load_train=False, load_val=False, load_test=True
                    )
                    test_dataset = _keep_audio_and_id(test_ls, target_sr=target_sr)

                return train_dataset, val_dataset, test_dataset

            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        if stage in (None, 'fit'):
            if self.train_dataset is None or self.val_dataset is None:
                train_ds, val_ds, _ = _load_splits(load_train=True, load_val=True, load_test=False)
                self.train_dataset = train_ds
                self.val_dataset = add_vc_pair_in_test_dataset(
                    self.cfg, val_ds, enforce_min_ref_filter=False)
                try:
                    print(f"Train dataset size: {len(self.train_dataset)}")
                except TypeError:
                    print("Train dataset: streaming IterableDataset (size unknown).")
                print(f"Validation dataset size: {len(self.val_dataset)}")

        if stage in (None, 'validate'):
            if self.val_dataset is None:
                _, val_ds, _ = _load_splits(load_train=False, load_val=True, load_test=False)
                self.val_dataset = add_vc_pair_in_test_dataset(
                    self.cfg, val_ds, enforce_min_ref_filter=False)
                print(f"Validation dataset size: {len(self.val_dataset)}")

        if stage in (None, 'test'):
            if self.test_dataset is None:
                _, _, test_ds = _load_splits(load_train=False, load_val=False, load_test=True)
                print("[VCTK] setup: test_ds loaded from _load_splits")
                if self.dataset_name == "vctk":
                    vctk_cfg = self.cfg.dataset.get("vctk", {})
                    utts_per_speaker = int(vctk_cfg.get("utts_per_speaker", 10))
                    refs_per_utt = int(vctk_cfg.get("refs_per_utt", 10))
                    max_speakers = vctk_cfg.get("max_speakers", None)
                    if max_speakers is not None:
                        try:
                            max_speakers = int(max_speakers)
                        except Exception:
                            max_speakers = None

                    seed = int(getattr(self.cfg.dataset, "pair_seed", 42))
                    print("[VCTK] setup: calling _sample_vctk_source_indices")
                    src_indices = _sample_vctk_source_indices(
                        test_ds,
                        seed=seed,
                        utts_per_speaker=utts_per_speaker,
                        max_speakers=max_speakers,
                    )
                    print(f"[VCTK] setup: _sample_vctk_source_indices returned {len(src_indices)} indices")
                    # In "same" mode: evaluate only source utterances once.
                    if str(getattr(self.cfg, "voice_conversion", "same")).lower() != "vc":
                        self.test_dataset = test_ds.select(src_indices)
                        print(f"VCTK test (same) size: {len(self.test_dataset)}")
                    else:
                        enforce = True
                        min_ref_seconds = float(getattr(self.cfg.dataset, "min_ref_seconds", 0.0))
                        print(f"[VCTK] setup: calling _build_vctk_vc_pairs (srcs={len(src_indices)}, refs_per_utt={refs_per_utt})")
                        src_rep, ref_list = _build_vctk_vc_pairs(
                            test_ds,
                            seed=seed,
                            source_base_indices=src_indices,
                            refs_per_source=refs_per_utt,
                            enforce_min_ref_filter=enforce,
                            min_ref_seconds=min_ref_seconds,
                        )
                        self.test_dataset = VctkVCEvalDataset(test_ds, src_rep, ref_list)
                        print("[VCTK] setup: VctkVCEvalDataset created")
                        # informative logging: speaker count in selected pool
                        try:
                            uniq_spk = sorted(set([str(s) for s in test_ds["speaker_id"]]))
                            print(f"VCTK speakers (post-filter): {len(uniq_spk)}")
                        except Exception:
                            pass
                        print(f"VCTK test (vc) size: {len(self.test_dataset)} (src={len(src_indices)}, refs_per_utt={refs_per_utt})")
                else:
                    self.test_dataset = add_vc_pair_in_test_dataset(
                        self.cfg, test_ds, enforce_min_ref_filter=True)
                    print(f"Test dataset size: {len(self.test_dataset)}")

    def get_loader(self, phase, dataset):
        phase_cfg = self.cfg.dataset.get(phase)
        batch_size = phase_cfg.batch_size
        num_workers = int(getattr(phase_cfg, "num_workers", 0))
        prefetch_factor = int(getattr(phase_cfg, "prefetch_factor", 2)) if num_workers > 0 else None
        persistent_workers = getattr(phase_cfg, "persistent_workers", False)
        if isinstance(persistent_workers, str):
            persistent_workers = persistent_workers.lower() in {"1", "true", "yes", "on"}
        else:
            persistent_workers = bool(persistent_workers)
        multiprocessing_context = getattr(phase_cfg, "multiprocessing_context", None) if num_workers > 0 else None
        if multiprocessing_context in ("", "none", "None"):
            multiprocessing_context = None
        if isinstance(dataset, IterableDataset):
            # Streaming path: dataset already yields final tensors.
            mls_cfg = self.cfg.dataset.get("mls", {})
            stream_num_workers = int(mls_cfg.get("streaming_num_workers", 0)) if hasattr(mls_cfg, "get") else int(getattr(mls_cfg, "streaming_num_workers", 0))
            dl = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=stream_num_workers,
                prefetch_factor=(phase_cfg.prefetch_factor if stream_num_workers > 0 else None),
                collate_fn=_collate_bicodec_streaming,
                persistent_workers=(stream_num_workers > 0),
                drop_last=bool(getattr(phase_cfg, "shuffle", False)),
                pin_memory=torch.cuda.is_available(),
            )
        else:
            ds = FSDataset(phase, self.cfg, dataset)
            # Train phase: use ResumableSampler for seamless mid-epoch resume
            sampler = None
            shuffle = phase_cfg.shuffle
            use_resumable_sampler = bool(getattr(phase_cfg, "use_resumable_sampler", True))
            if phase == "train":
                if use_resumable_sampler:
                    seed = int(getattr(self.cfg, "seed", 1024))
                    if dist.is_available() and dist.is_initialized():
                        sampler = ResumableDistributedSampler(
                            dataset=ds,
                            num_replicas=dist.get_world_size(),
                            rank=dist.get_rank(),
                            shuffle=phase_cfg.shuffle,
                            seed=seed,
                            drop_last=phase_cfg.shuffle,
                        )
                    else:
                        sampler = ResumableRandomSampler(
                            data_source=ds,
                            shuffle=phase_cfg.shuffle,
                            seed=seed,
                        )
                    shuffle = False  # must be False when using a sampler
                    # Restore sampler on resume: saved state first, else derive from total_step
                    # (not Lightning global_step, which is 2x under manual gen/disc optimizers).
                    if getattr(self, "trainer", None) and getattr(self.trainer, "ckpt_path", None):
                        ckpt_path = self.trainer.ckpt_path
                        if ckpt_path and os.path.isfile(str(ckpt_path)):
                            try:
                                ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                                accum = _manual_gradient_accumulation_steps(self.cfg)
                                _restore_train_sampler_from_ckpt(
                                    sampler,
                                    ckpt,
                                    batch_size=batch_size,
                                    dataset=dataset,
                                    phase_cfg=phase_cfg,
                                    accum=accum,
                                )
                            except Exception as e:
                                logger.warning("Could not restore sampler state: %s", e)
                    self._train_sampler = sampler
                    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
                        world = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
                        eff_batch = int(batch_size) * int(world) * _manual_gradient_accumulation_steps(self.cfg)
                        batches_per_epoch = _sampler_batches_per_epoch(
                            dataset_len=len(ds),
                            batch_size=batch_size,
                            shuffle=bool(phase_cfg.shuffle),
                            num_replicas=world,
                        )
                        print(
                            colored(
                                f"[sampler] {type(sampler).__name__}: world={world}, "
                                f"per_device_batch={batch_size}, accum={_manual_gradient_accumulation_steps(self.cfg)}, "
                                f"effective_batch={eff_batch}, batches_per_epoch={batches_per_epoch}",
                                "cyan",
                            )
                        )
                    _verify_distributed_sampler_shards(sampler)
                else:
                    sampler = None
                    shuffle = phase_cfg.shuffle
                    self._train_sampler = None
            elif hasattr(self, "_train_sampler"):
                pass  # keep existing _train_sampler from train phase

            dl = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                collate_fn=ds.collate_fn,
                persistent_workers=persistent_workers,
                multiprocessing_context=multiprocessing_context,
                drop_last=phase_cfg.shuffle,
                pin_memory=torch.cuda.is_available(),
            )
        return dl

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup('fit')
        return self.get_loader('train', self.train_dataset)

    def val_dataloader(self):
        if self.val_dataset is None:
            self.setup('validate')
        return self.get_loader('val', self.val_dataset)

    def test_dataloader(self):
        if self.test_dataset is None:
            self.setup('test')
        if self.dataset_name == "vctk":
            print("[VCTK] test_dataloader: creating loader")
        return self.get_loader('test', self.test_dataset)


class FSDataset(Dataset):
    """FastSpeech dataset batching text, mel, pitch 
    and other acoustic features

    Args:
        phase: train, val, test
        cfg: hydra config
    """

    def __init__(self, phase, cfg, dataset):
        self.phase = phase
        self.cfg = cfg
        # self.phase_cfg = cfg.dataset.get(phase)
        self.dataset = dataset
        self.dataset_name = cfg.dataset.get('name', 'librispeech').lower()
        # NOTE:
        # Historical configs bypassed `get_ref_clip` in TRAIN for all/non-streaming
        # MLS and used `ref_wav == target_wav`. Keep that behavior unless a config
        # explicitly chooses another train reference mode.
        mls_cfg = cfg.dataset.get("mls", {})
        try:
            mls_streaming = bool(mls_cfg.get("streaming", False))
        except Exception:
            mls_streaming = bool(getattr(mls_cfg, "streaming", False))
        legacy_skips_train_ref_clip = (
            (self.phase == "train")
            and (
                (self.dataset_name == "all")
                or (self.dataset_name == "mls" and (not mls_streaming))
            )
        )
        configured_ref_mode = cfg.dataset.get("train_ref_clip_mode", None)
        if configured_ref_mode is None:
            self.train_ref_clip_mode = "target" if legacy_skips_train_ref_clip else "random"
        else:
            self.train_ref_clip_mode = str(configured_ref_mode).strip().lower().replace("-", "_")
        if self.train_ref_clip_mode in {"get_ref_clip", "random_crop"}:
            self.train_ref_clip_mode = "random"
        if self.train_ref_clip_mode in {"same", "target_wav"}:
            self.train_ref_clip_mode = "target"
        if self.train_ref_clip_mode in {"near", "nearby_nonoverlap", "nearby_non_overlap"}:
            self.train_ref_clip_mode = "nearby"
        if self.train_ref_clip_mode in {"same_context", "target_context_ref", "target_inclusive"}:
            self.train_ref_clip_mode = "target_context"
        valid_ref_modes = {"random", "target", "nearby", "target_context"}
        if self.train_ref_clip_mode not in valid_ref_modes:
            raise ValueError(
                f"Unsupported dataset.train_ref_clip_mode={configured_ref_mode!r}; "
                f"expected one of {sorted(valid_ref_modes)}"
            )
        self.skip_ref_clip_train = self.phase == "train" and self.train_ref_clip_mode == "target"
        if self.skip_ref_clip_train:
            assert cfg.preprocess.audio.sr == 16000, "Only 16kHz is supported for skipping ref-clip in train"
        if self.phase == "train":
            print(
                colored(
                    f"Train reference clip mode for {self.dataset_name}: {self.train_ref_clip_mode}",
                    "green",
                    attrs=['bold'],
                )
            )
        self.ocwd = _original_cwd()

        self.sr = cfg.preprocess.audio.sr
        # self.filelist = utils.read_filelist(join(self.ocwd, self.phase_cfg.filelist))
        self.min_audio_length = cfg.dataset.min_audio_length
        self.latent_hop_length = cfg.dataset.latent_hop_length
        self.ref_segment_duration = cfg.dataset.ref_segment_duration

    def __len__(self):
        return len(self.dataset)

    def _get_pair_item(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a paired/reference example.

        - For normal HF `Dataset`: uses random-access indexing.
        - For wrapped evaluation datasets (e.g. `VctkVCEvalDataset`): use `get_base_item`
          so `vc_pair_idx` can refer to base-dataset indices.
        """
        if hasattr(self.dataset, "get_base_item"):
            return getattr(self.dataset, "get_base_item")(int(idx))
        return self.dataset[int(idx)]

    def load_wav(self, path):
        wav, sr = librosa.load(path, sr=self.sr)
        return wav

    def get_ref_clip(self, wav: torch.Tensor, target_start: int, target_end: int, use_full: bool) -> torch.Tensor:
        ref_len = (int(self.sr * self.ref_segment_duration) //
                   self.latent_hop_length) * self.latent_hop_length
        wav_len = wav.shape[0]
        seg_len = target_end - target_start

        if not use_full:
            # 큰 ones_like 생성/곱 대신 복제 후 슬라이스 0 대입
            ref = wav.clone()
            if self.phase == 'train' and seg_len > 0 and (wav_len / max(seg_len, 1) > 2):
                ref[target_start:target_end] = 0
        else:
            ref = wav

        # random crop or repeat
        if ref_len > wav_len:
            reps = (ref_len + wav_len - 1) // wav_len
            ref = ref.repeat(reps)
        if ref.shape[0] > ref_len:
            start = random.randint(
                0, ref.shape[0] - ref_len) if self.phase == 'train' else 0
            ref = ref[start:start+ref_len]
        return ref[:ref_len]

    def get_nearby_ref_clip(self, wav: torch.Tensor, target_start: int, target_end: int) -> torch.Tensor:
        ref_len = (int(self.sr * self.ref_segment_duration) //
                   self.latent_hop_length) * self.latent_hop_length
        wav_len = wav.shape[0]
        if wav_len <= 0 or ref_len <= 0:
            return wav

        target_start = max(0, min(int(target_start), wav_len))
        target_end = max(target_start, min(int(target_end), wav_len))

        full_candidates = []
        if target_start >= ref_len:
            full_candidates.append((target_start - ref_len, target_start))
        if wav_len - target_end >= ref_len:
            full_candidates.append((target_end, target_end + ref_len))
        if full_candidates:
            start, end = random.choice(full_candidates) if self.phase == 'train' else full_candidates[0]
            return wav[start:end]

        left = wav[max(0, target_start - ref_len):target_start]
        right = wav[target_end:min(wav_len, target_end + ref_len)]

        if left.numel() > 0 and right.numel() > 0:
            left_take = min(left.numel(), ref_len // 2)
            right_take = min(right.numel(), ref_len - left_take)
            remaining = ref_len - left_take - right_take
            if remaining > 0:
                extra_left = min(left.numel() - left_take, remaining)
                left_take += extra_left
                remaining -= extra_left
            if remaining > 0:
                right_take += min(right.numel() - right_take, remaining)
            chunks = []
            if left_take > 0:
                chunks.append(left[-left_take:])
            if right_take > 0:
                chunks.append(right[:right_take])
            ref = torch.cat(chunks, dim=0)
        elif left.numel() > 0:
            ref = left[-ref_len:]
        elif right.numel() > 0:
            ref = right[:ref_len]
        else:
            return wav.new_zeros(ref_len)

        if ref.shape[0] < ref_len:
            reps = (ref_len + ref.shape[0] - 1) // ref.shape[0]
            ref = ref.repeat(reps)
        return ref[:ref_len]

    def get_target_context_ref_clip(self, wav: torch.Tensor, target_start: int, target_end: int) -> torch.Tensor:
        ref_len = (int(self.sr * self.ref_segment_duration) //
                   self.latent_hop_length) * self.latent_hop_length
        wav_len = wav.shape[0]
        if wav_len <= 0 or ref_len <= 0:
            return wav

        target_start = max(0, min(int(target_start), wav_len))
        target_end = max(target_start, min(int(target_end), wav_len))
        target_len = target_end - target_start

        if wav_len <= ref_len:
            reps = (ref_len + wav_len - 1) // wav_len
            return wav.repeat(reps)[:ref_len]

        if target_len <= 0:
            return wav[:ref_len]

        if target_len >= ref_len:
            start = target_start + (target_len - ref_len) // 2
            return wav[start:start + ref_len]

        extra = ref_len - target_len
        left_avail = target_start
        right_avail = wav_len - target_end
        left_take = min(left_avail, extra // 2)
        right_take = min(right_avail, extra - left_take)

        remaining = extra - left_take - right_take
        if remaining > 0:
            add_left = min(left_avail - left_take, remaining)
            left_take += add_left
            remaining -= add_left
        if remaining > 0:
            right_take += min(right_avail - right_take, remaining)

        start = target_start - left_take
        end = target_end + right_take
        return wav[start:end][:ref_len]

    # @timeit
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # numpy -> torch는 복사 없는 from_numpy 선호
        wav = torch.from_numpy(item['audio']['array'])
        if wav.dtype != torch.float32:
            wav = wav.float()

        if self.sr == 24000:  # typically 50 Hz (480 samples hop)
            assert self.latent_hop_length == 480, (
                f"24kHz expects latent_hop_length=480 (50Hz), got {self.latent_hop_length}"
            )
            assert self.min_audio_length >= self.latent_hop_length and self.min_audio_length % self.latent_hop_length == 0, (
                f"min_audio_length must be a positive multiple of latent_hop_length ({self.latent_hop_length}), got {self.min_audio_length}"
            )
        elif self.sr == 16000:  # 100 Hz (160) or 50 Hz (320) are both common
            assert self.latent_hop_length in (160, 320), (
                f"16kHz expects latent_hop_length=160(100Hz) or 320(50Hz), got {self.latent_hop_length}"
            )
            assert self.min_audio_length >= self.latent_hop_length and self.min_audio_length % self.latent_hop_length == 0, (
                f"min_audio_length must be a positive multiple of latent_hop_length ({self.latent_hop_length}), got {self.min_audio_length}"
            )
        else:
            raise ValueError(f"Unsupported sample rate: {self.sr}")

        # 길이 정렬 (latent_hop_length * 4 배수)
        if self.latent_hop_length:
            base_frames = (wav.shape[0] // self.latent_hop_length // 4) * 4
            wav = wav[: base_frames * self.latent_hop_length]

        full_wav = wav
        wav_len = wav.shape[0]

        # 세그먼트 길이 (hop 정렬)
        seg_len = int(self.min_audio_length)
        if self.latent_hop_length:
            seg_len = max(self.latent_hop_length,
                          (seg_len // self.latent_hop_length) * self.latent_hop_length)

        is_test = (self.phase == 'test')
        use_full = (self.phase in ['test', 'predict'])
        ref_src_16k = None  # test에서 _ref.wav 저장용 (get_ref_clip 이전 raw)

        # 평가에서는 전체 사용 + hop 정렬, 학습/검증에서는 부족 시 pad
        if use_full and self.latent_hop_length:
            wav = wav[: (wav_len // self.latent_hop_length)
                      * self.latent_hop_length]
            wav_len = wav.shape[0]
            seg_len = wav_len
        elif not use_full and wav_len < seg_len:
            wav = F.pad(wav, (0, seg_len - wav_len))
            full_wav = wav
            wav_len = wav.shape[0]

        # 타겟 오디오 선택
        if use_full:
            start, end = 0, wav_len
            target_wav = wav
        else:
            if self.phase in ['train', 'val'] and wav_len > seg_len:
                start_max = wav_len - seg_len
                if self.latent_hop_length:
                    start = random.randint(
                        0, start_max // self.latent_hop_length) * self.latent_hop_length
                else:
                    start = random.randint(0, start_max)
            else:
                start = 0
            end = start + seg_len
            target_wav = wav[start:end]

        is_24k = int(self.sr) == 24000

        # 24k -> 16k 보조 함수 (SdpCodec speaker/SSL branches expect 16k)
        def to16k(x: torch.Tensor) -> torch.Tensor:
            if not is_24k:
                return x.clamp(-1.0, 1.0)
            y = torchaudio.functional.resample(x, int(self.sr), 16000)
            return y.clamp(-1.0, 1.0)

        target_id = None
        is_val = (self.phase == 'val')

        # 참조 오디오 (LibriTTS인 경우 16k)
        # Validation: prepare both same-speaker and VC references
        if is_val:
            # Same speaker reference
            ref_wav_same = self.get_ref_clip(full_wav, start, end, use_full)
            ref_wav_same_16k = to16k(ref_wav_same) if is_24k else ref_wav_same

            # VC reference (different speaker)
            vc_pair_idx = int(item['vc_pair_idx'])
            vc_paired = self._get_pair_item(vc_pair_idx)
            vc_ref_src = torch.from_numpy(
                vc_paired['audio']['array']).to(wav.dtype)
            if is_24k:
                # 24 kHz input: crop at 24 kHz, then downsample to 16 kHz
                vc_ref_24k = self.get_ref_clip(
                    vc_ref_src, start, end, use_full)
                ref_wav_vc_16k = to16k(vc_ref_24k)
            else:
                # 16 kHz input: crop directly at 16 kHz
                ref_wav_vc_16k = self.get_ref_clip(
                    vc_ref_src, start, end, use_full)
            vc_target_id = vc_paired['id']

            ref_wav_16k = ref_wav_same_16k  # Default for backward compatibility
        elif is_test and self.cfg.voice_conversion == 'vc':
            pair_idx = int(item['vc_pair_idx'])
            paired = self._get_pair_item(pair_idx)
            print(
                f'Voice conversion {self.cfg.voice_conversion}: {paired["id"]} -> {item["id"]}')
            ref_src = torch.from_numpy(paired['audio']['array']).to(wav.dtype)
            # 저장용 raw ref_src (get_ref_clip 이전)
            ref_src_16k = to16k(ref_src) if is_24k else ref_src.clamp(-1.0, 1.0)
            if is_24k:
                # 24 kHz input: crop at 24 kHz, then downsample to 16 kHz
                ref_24k = self.get_ref_clip(ref_src, start, end, use_full)
                ref_wav_16k = to16k(ref_24k)
            else:
                # 16 kHz input: crop directly at 16 kHz
                ref_wav_16k = self.get_ref_clip(ref_src, start, end, use_full)
            target_id = paired['id']
        else:
            if is_test:
                print(f'Reconstruction (SAME): {item["id"]} -> {item["id"]}')
            if self.phase == 'train' and self.train_ref_clip_mode == "target":
                # Train-only option (16 kHz only; asserted in __init__):
                # use target segment itself as reference to remove ref-clip variance.
                ref_wav_16k = target_wav
            elif self.phase == 'train' and self.train_ref_clip_mode == "nearby":
                ref_wav = self.get_nearby_ref_clip(full_wav, start, end)
                ref_wav_16k = to16k(ref_wav) if is_24k else ref_wav
            elif self.phase == 'train' and self.train_ref_clip_mode == "target_context":
                ref_wav = self.get_target_context_ref_clip(full_wav, start, end)
                ref_wav_16k = to16k(ref_wav) if is_24k else ref_wav
            else:
                ref_wav = self.get_ref_clip(full_wav, start, end, use_full)
                ref_wav_16k = to16k(ref_wav) if is_24k else ref_wav

            # 저장용 raw ref_src는 동일 샘플의 full_wav (get_ref_clip 이전)
            if is_test:
                ref_src_16k = to16k(full_wav) if is_24k else full_wav.clamp(-1.0, 1.0)

        # 출력 오디오 구성
        # - `wav`: always 16k (for SSL/speaker encoder branches)
        # - `wav_24k`: present only when model/preprocess is 24k
        if is_24k:
            wav_24k = target_wav
            wav_16k = to16k(wav_24k)

        # Final safety clamp for target waveform
        target_wav = target_wav.clamp(-1.0, 1.0)

        out = {
            'fid': item['id'],
            'wav': wav_16k if is_24k else target_wav,
            'wav_24k': wav_24k if is_24k else None,
            'ref_wav': ref_wav_16k,
            'target_id': target_id if is_test and self.cfg.voice_conversion == 'vc' else None,
            'ref_src': ref_src_16k if is_test else None,
        }

        # Validation: add VC reference
        if is_val:
            out['ref_wav_vc'] = ref_wav_vc_16k
            out['vc_target_id'] = vc_target_id

        return out

    def collate_fn(self, bs):
        fids = [b['fid'] for b in bs]
        wavs = [b['wav'] for b in bs]
        if int(self.sr) == 24000:
            wav_24ks = [b['wav_24k'] for b in bs]
            wav_24ks = torch.stack(wav_24ks)
        else:
            wav_24ks = None
        ref_wavs = [b['ref_wav'] for b in bs]
        if self.phase == 'test':
            target_ids = [b['target_id'] for b in bs]
            ref_srcs = [b.get('ref_src', None) for b in bs]
        if self.phase == 'val':
            ref_wavs_vc = [b['ref_wav_vc'] for b in bs]
            vc_target_ids = [b['vc_target_id'] for b in bs]
        if self.latent_hop_length and self.phase == 'test':
            max_len = max([w.shape[0] for w in wavs])
            max_len = (max_len // self.latent_hop_length) * \
                self.latent_hop_length
            wavs = [F.pad(w, (0, max_len - w.shape[0])) for w in wavs]

        wavs = torch.stack(wavs)
        ref_wavs = torch.stack(ref_wavs)

        # Safety clamp after batching
        wavs = wavs.clamp(-1.0, 1.0)
        ref_wavs = ref_wavs.clamp(-1.0, 1.0)

        if self.phase == 'test':
            if any(r is not None for r in ref_srcs):
                # 길이가 다르면 max_len에 pad 후 stack
                lens = [int(r.shape[0]) for r in ref_srcs if isinstance(r, torch.Tensor)]
                max_ref_len = max(lens) if len(lens) > 0 else 0
                ref_srcs = [
                    (F.pad(r, (0, max_ref_len - int(r.shape[0]))) if isinstance(r, torch.Tensor) and int(r.shape[0]) < max_ref_len else r)
                    for r in ref_srcs
                ]
                ref_srcs = torch.stack(ref_srcs).clamp(-1.0, 1.0)
            else:
                ref_srcs = None

        out = {
            'fid': fids,
            'wav': wavs,
            'wav_24k': wav_24ks,
            'ref_wav': ref_wavs,
            'target_id': target_ids if self.phase == 'test' else None,
            'ref_src': ref_srcs if self.phase == 'test' else None,
        }

        if self.phase == 'val':
            ref_wavs_vc = torch.stack(ref_wavs_vc)
            ref_wavs_vc = ref_wavs_vc.clamp(-1.0, 1.0)
            out['ref_wav_vc'] = ref_wavs_vc
            out['vc_target_id'] = vc_target_ids

        return out
