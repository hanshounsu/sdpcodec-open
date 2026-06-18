#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-sdpcodec:latest}"
CONDA_ENV="${CONDA_ENV:-sdpcodec}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/${CONDA_ENV}/bin/python}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
UID_GID="$(id -u):$(id -g)"

mkdir -p "${HOME}/.cache/matplotlib" "${HOME}/.cache/fontconfig"

VOLUME_ARGS=(
  -v "${HOME}:${HOME}"
)
if [[ -d /data ]]; then
  VOLUME_ARGS+=(-v /data:/data)
fi

exec docker run --rm --init \
  --user "${UID_GID}" \
  -e HOME="${HOME}" \
  -e MPLCONFIGDIR="${HOME}/.cache/matplotlib" \
  -e XDG_CACHE_HOME="${HOME}/.cache" \
  "${VOLUME_ARGS[@]}" \
  -w "${REPO_ROOT}" \
  "${IMAGE}" \
  "${PYTHON_BIN}" -u -m sdpcodec.train "$@"
