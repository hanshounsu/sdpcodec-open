# Pretrained Assets

This repository keeps code and small assets in git, but large third-party model
weights should be downloaded by each user. The paths below cover the
VQ-Wav2Vec reproduction config in `configs/sdpcodec_vqw2v_rvq300.yaml`.

## Required For Speaker Conditioning

### WavLM Large speaker encoder

SDPCodec uses WavLM Large for speaker conditioning. Download `WavLM-Large.pt`
from the official Microsoft WavLM release page:

- Microsoft UniLM WavLM README: https://github.com/microsoft/unilm/tree/master/wavlm
- Hugging Face mirror, useful for reference: https://huggingface.co/microsoft/wavlm-large

Place the file here:

```text
pretrained_models/wavlm/WavLM-Large.pt
```

The expected layout is:

```text
sdpcodec-open/
  pretrained_models/
    wavlm/
      WavLM-Large.pt
```

If you prefer a shared model directory outside the repository, keep the file
there and override the config:

```bash
python -m sdpcodec.train \
  model.speaker_encoder.wavlm_checkpoint=/models/wavlm/WavLM-Large.pt
```

The same override works for inference:

```bash
python -m sdpcodec.infer \
  --checkpoint /path/to/sdpcodec.ckpt \
  --source /path/to/source.wav \
  --output outputs/infer/source_rec.wav \
  model.speaker_encoder.wavlm_checkpoint=/models/wavlm/WavLM-Large.pt
```

## Required For The Main VQ-Wav2Vec Reproduction Config

The main reproduction config is:

```text
configs/sdpcodec_vqw2v_rvq300.yaml
```

It uses the pretrained fairseq VQ-Wav2Vec k-means checkpoint as the frozen
content encoder. The expected local path is:

```text
pretrained_models/vq_wav2vec/vq-wav2vec_kmeans.pt
```

The expected layout is:

```text
sdpcodec-open/
  pretrained_models/
    vq_wav2vec/
      vq-wav2vec_kmeans.pt
    wavlm/
      WavLM-Large.pt
```

The checkpoint is the standard S3PRL/fairseq upstream named:

```text
vq_wav2vec_kmeans
```

Common ways to prepare it:

1. Let S3PRL download/cache the upstream, then run with:

   ```bash
   python -m sdpcodec.train \
     model.codec_encoder.vqw2v_checkpoint=null
   ```

2. Put a local fairseq checkpoint at
   `pretrained_models/vq_wav2vec/vq-wav2vec_kmeans.pt` and use the config as-is.

3. Keep the checkpoint in a shared model directory and override:

   ```bash
   python -m sdpcodec.train \
     model.codec_encoder.vqw2v_checkpoint=/models/vq_wav2vec/vq-wav2vec_kmeans.pt
   ```

Install the optional dependencies before using this config:

```bash
python -m pip install -r requirements-vqw2v.txt
```

If you rely on S3PRL downloads on a cluster, set a persistent cache directory:

```bash
export S3PRL_CACHE_ROOT=/path/to/s3prl-cache
```

## Hugging Face Cache

The dataset loader and optional WER validation use Hugging Face caches. Useful
cache environment variables:

```bash
export HF_HOME=/path/to/huggingface-cache
export TRANSFORMERS_CACHE=/path/to/huggingface-cache/transformers
export HF_DATASETS_CACHE=/path/to/huggingface-cache/datasets
```

For shared clusters, set these to a persistent filesystem so every run does not
download the same files again.

## Bundled In This Repository

### FCPE pitch estimator

The F0 path uses FCPE through the upstream FCPE git submodule. The submodule
includes the small FCPE checkpoint:

```text
pretrained_models/pitch_estimator/FCPE/torchfcpe/assets/fcpe_c_v001.pt
```

Initialize submodules after cloning:

```bash
git submodule update --init --recursive
```

The upstream project is:

- https://github.com/CNChTu/FCPE
- https://pypi.org/project/torchfcpe/

## Optional Validation Models

The public config disables heavy validation metrics by default:

```yaml
train:
  use_val_utmos: false
  use_val_wer: false
```

You can enable them when you want extra monitoring and your environment can
download the metric models.

### UTMOS

Enable:

```bash
python -m sdpcodec.train train.use_val_utmos=true
```

The code loads UTMOS with `torch.hub` from:

```text
tarepan/SpeechMOS:v1.2.0
```

Reference:

- https://github.com/tarepan/SpeechMOS

Set a persistent torch cache if you are on a cluster:

```bash
export TORCH_HOME=/path/to/torch-cache
```

### WER with HuBERT CTC

Enable:

```bash
python -m sdpcodec.train train.use_val_wer=true
```

By default this uses the same Hugging Face model id:

```text
facebook/hubert-large-ls960-ft
```

Override it if needed:

```bash
python -m sdpcodec.train \
  train.use_val_wer=true \
  train.val_hubert_model=facebook/hubert-large-ls960-ft
```

## Offline Or Firewalled Machines

For machines without internet access:

1. Download `WavLM-Large.pt` on a connected machine and copy it to
   `pretrained_models/wavlm/WavLM-Large.pt`, or pass
   `model.speaker_encoder.wavlm_checkpoint=/path/to/WavLM-Large.pt`.
2. For the VQ-Wav2Vec config, copy `vq-wav2vec_kmeans.pt` to
   `pretrained_models/vq_wav2vec/vq-wav2vec_kmeans.pt`, or pass
   `model.codec_encoder.vqw2v_checkpoint=/path/to/vq-wav2vec_kmeans.pt`.
3. If you need optional WER validation, pre-cache the HuBERT CTC model on a
   connected machine:

   ```bash
   HF_HOME=/path/to/hf-cache python - <<'PY'
   from transformers import AutoProcessor, HubertForCTC

   AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
   HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
   PY
   ```

4. Copy that cache directory to the offline machine and set:

   ```bash
   export HF_HOME=/path/to/hf-cache
   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   export HF_DATASETS_OFFLINE=1
   ```

5. Leave optional validation metrics disabled unless their `torch.hub` cache has
   also been prepared.

## Common Path Checks

From the repository root:

```bash
test -f pretrained_models/wavlm/WavLM-Large.pt && echo "WavLM found"
test -f pretrained_models/vq_wav2vec/vq-wav2vec_kmeans.pt && echo "VQ-Wav2Vec found"
test -f pretrained_models/pitch_estimator/FCPE/torchfcpe/assets/fcpe_c_v001.pt && echo "FCPE found"
```

If WavLM is missing, training will fail while constructing the speaker encoder.
If optional WER is enabled and HuBERT CTC is not cached on a machine without
internet, validation WER will be disabled after the load failure.
