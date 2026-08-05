#!/usr/bin/env bash
#
# Start an interactive container with the repository bind-mounted.
#
#   ./docker/run.sh          # every GPU the daemon exposes
#   ./docker/run.sh 0        # GPU 0 only
#   ./docker/run.sh 0,1      # GPUs 0 and 1
#
# The no-argument form is what CLAUDE.md documents, and it used to abort
# immediately: `set -eu` plus a bare `DEVICE=$1` gave
# "run.sh: line 10: $1: unbound variable". With no argument the container now
# sees every GPU `--gpus all` exposes and CUDA_VISIBLE_DEVICES is left unset.
#
# HOST PATHS. The mounts below were hardcoded to the FIIT cluster's layout, which
# no stranger has. Each is now an environment variable with the cluster value as
# its default, so the script runs elsewhere without being edited:
#
#   DP_HOST_DATASETS  -> /mnt/datasets and ${CONTAINER_WORKSPACE}/.datasets
#   DP_HOST_SCRATCH   -> ${CONTAINER_WORKSPACE}/.scratch
#   DP_HOST_NFS       -> itself, and skipped when it does not exist
#   CONTAINER_WORKSPACE, IMAGE_TAG
#
# Inside the container run `pip install -e . --no-deps` once: the image ships
# dependencies only, and the editable install is what provides dp-train and the
# other five console scripts now that PYTHONPATH is no longer baked in.

# Fail on error and unset variables.
set -eu -o pipefail

CWD=$(readlink -e "$(dirname "$0")")
cd "${CWD}/.." || exit $?
source ./docker/common.sh

DEVICE="${1:-}"
if [[ -n "${DEVICE}" ]]; then
    DEVICE_NAME=$(echo "${DEVICE}" | tr "," "-")
    echo "Using GPU devices: ${DEVICE}"
else
    DEVICE_NAME="all"
    echo "Using GPU devices: all (CUDA_VISIBLE_DEVICES not set)"
fi

USER_NAME="${USER:-$(whoami)}"
export USER_NAME
echo "User: ${USER_NAME}"

CONTAINER_WORKSPACE="${CONTAINER_WORKSPACE:-/workspace/${PROJECT_NAME}}"
DP_HOST_DATASETS="${DP_HOST_DATASETS:-/mnt/scratch/${USER_NAME}/.datasets}"
DP_HOST_SCRATCH="${DP_HOST_SCRATCH:-/mnt/scratch/${USER_NAME}/${PROJECT_NAME}}"
DP_HOST_NFS="${DP_HOST_NFS:-/mnt/nfs-data}"

MOUNTS=(
    -v "${CWD}/..":"${CONTAINER_WORKSPACE}"
    -v "${DP_HOST_DATASETS}":/mnt/datasets
    -v "${DP_HOST_DATASETS}":"${CONTAINER_WORKSPACE}/.datasets"
    -v "${DP_HOST_SCRATCH}":"${CONTAINER_WORKSPACE}/.scratch"
)
# Bind-mounting a host path that does not exist makes docker create it as an
# empty root-owned directory, which then looks like an empty dataset rather than
# a missing one. Only mount what is actually there.
if [[ -d "${DP_HOST_NFS}" ]]; then
    MOUNTS+=(-v "${DP_HOST_NFS}":"${DP_HOST_NFS}")
else
    echo "note: ${DP_HOST_NFS} does not exist on this host; not mounting it." >&2
fi
# The W&B API key lives here. Mounted from the host, never baked into the image.
if [[ -f "${HOME}/.netrc" ]]; then
    MOUNTS+=(-v "${HOME}/.netrc":/root/.netrc)
else
    echo "note: ${HOME}/.netrc not found; W&B logging will need WANDB_API_KEY." >&2
fi

ENV_ARGS=()
if [[ -n "${DEVICE}" ]]; then
    ENV_ARGS+=(-e CUDA_VISIBLE_DEVICES="${DEVICE}")
fi

docker run \
    -it --rm \
    --name "${PROJECT_NAME}-${DEVICE_NAME}" \
    --gpus all \
    --privileged \
    --shm-size 8g \
    --device /dev/fuse \
    "${MOUNTS[@]}" \
    ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} \
    -w "${CONTAINER_WORKSPACE}" \
    "${IMAGE_TAG}"
