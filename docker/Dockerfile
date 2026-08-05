# syntax=docker/dockerfile:1
#
# Dependency image for dp-code. It carries the Python environment and NOTHING
# ELSE: no source is baked in, because docker/run.sh bind-mounts the repository
# over ${WORKSPACE_DIR} at run time and would shadow any copy. Consequently the
# image alone reproduces nothing — it is half of the environment, and the clone
# is the other half.
#
# NOT VERIFIED TO BUILD. Four independent external dependencies gate it and none
# was exercised while this file was written (building was out of scope):
#   1. the base image tag below, a FIIT VGG-group image on Docker Hub;
#   2. https://deb.nodesource.com/setup_24.x, an unpinned installer script;
#   3. https://claude.ai/install.sh, likewise;
#   4. the smooth-topk GitHub commit pinned in requirements.txt, which needs git
#      and network at pip time.
# Treat a successful build as good news rather than as the documented path, and
# use the venv route (`pip install -e '.[dev]'`) if it fails.
#
# WHAT IS UNKNOWN ABOUT THE BASE IMAGE: it appears to supply /opt/venv and
# Python 3.10.11 (that is what the running container reports), but nothing here
# creates or activates a virtualenv, so `pip` at the install step lands in
# whichever interpreter the base image puts first on PATH. If you replace the
# base image, check that first.

ARG BASE_IMAGE=vggfiit/vgg-torch:2.0.0
FROM ${BASE_IMAGE}

# Where dependency manifests are copied during the build, and where the repository
# is expected to be mounted at run time. Both were hardcoded; they are build args
# now so the image can be rebuilt for a different mount point without editing this
# file. docker/run.sh mounts the clone at ${WORKSPACE_DIR}.
ARG SETUP_DIR=/setup
ARG PROJECT_NAME=dp-code
ARG WORKSPACE_DIR=/workspace/${PROJECT_NAME}

# Update the APT-GET
# `-y` is REQUIRED: a BuildKit build is non-interactive, so `apt-get install`
# without it aborts on the confirmation prompt unless the package happens to be
# present and current already. The two lines below lacked it.
RUN apt-get update --fix-missing
RUN apt-get install -y apt-transport-https
RUN apt-get install -y ca-certificates
RUN update-ca-certificates

# `make` is NOT in the base image (verified inside a running container: `make`
# is not found, and `apt-get install make` is what this line does about it). The
# documented development loop — `make install` / `make smoke` / `make test` /
# `make check-paths` / `make reference` — is unusable without it. `git` IS
# already present, which matters because the smooth-topk pin in requirements.txt
# is a git URL that pip must clone at install time.
RUN apt-get install -y --no-install-recommends make

RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash - \
    && apt-get update --fix-missing \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code in the container. Unpinned and unrelated to the science —
# it is a developer convenience, and it is the second of the two curl-piped
# installers noted at the top.
RUN curl -fsSL https://claude.ai/install.sh | bash \
    && echo 'export PATH="$HOME/.local/bin:$PATH"' >> /root/.bashrc
ENV PATH="/root/.local/bin:${PATH}"

# Install required packages.
# Both manifests are copied because requirements-dev.txt begins with
# `-r requirements.txt`, so one pip run installs the runtime pins and pytest.
# Nothing else is copied: see .dockerignore, without which this two-file COPY
# would ship a 327 GB build context.
WORKDIR ${SETUP_DIR}
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

# Working dir.
#
# PYTHONPATH IS DELIBERATELY NOT SET. An `ENV PYTHONPATH=${WORKSPACE_DIR}` line
# used to live here and was the ONLY mechanism that made `import project.survival`
# or `import tools.pam50_arms` resolve — outside this image every such import
# failed. `pip install -e .` now does that job, and it also installs the six
# console scripts (dp-train, dp-evaluate, dp-analysis, dp-data, dp-cptac,
# dp-config). Run it ONCE per container, after the bind mount is in place:
#
#     pip install -e . --no-deps
#
# `--no-deps` because the dependencies are already installed above, and editable
# because a non-editable install would copy `project` and `tools` into
# site-packages without their sibling data directories (dataset_csv/, splits/,
# tools/data/), which those modules reach by relative path.
WORKDIR ${WORKSPACE_DIR}
