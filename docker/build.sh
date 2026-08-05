#!/usr/bin/env bash
#
# Build the dependency image.
#
#   ./docker/build.sh                     # tag: dp-code
#   IMAGE_TAG=dp-code:cu117 ./docker/build.sh
#
# The build context is the repository root, and `.dockerignore` is what keeps
# that affordable: without it this ships ~327 GB (.datasets 231 GB, .scratch
# 95 GB, .git 1.4 GB) to the daemon in order to COPY two dependency manifests.
#
# The image has NEVER been verified to build — see the header of
# docker/Dockerfile for the four external dependencies that gate it.

# Fail on error and unset variables.
set -eu -o pipefail

CWD=$(readlink -e "$(dirname "$0")")
cd "${CWD}/.." || exit $?
source ./docker/common.sh

DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t "${IMAGE_TAG}" . || exit $?

cat <<EOF

Built ${IMAGE_TAG}. The image carries dependencies only: no source is baked in,
because docker/run.sh bind-mounts the clone over the working directory and would
shadow any copy. So install the package ONCE inside a running container:

    ./docker/run.sh 0
    pip install -e . --no-deps

That is what provides dp-train, dp-evaluate, dp-analysis, dp-data, dp-cptac and
dp-config, and it replaces the PYTHONPATH the image used to set. --no-deps
because the pins are already installed. It cannot be done during the build: the
egg-info an image-time install produces is hidden by the source bind mount.
EOF
