# syntax=docker/dockerfile:1

ARG CUDA_IMAGE=nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG TORCH_VERSION=2.12.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132
ARG OFOLDX_VERSION=0.0.0+docker

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/ofoldx-venv \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        hmmer \
        libgl1 \
        libglib2.0-0t64 \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv "${VIRTUAL_ENV}"

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"

WORKDIR /workspace/OFoldX

COPY pyproject.toml README.md LICENSE.md ./
COPY ofoldx ./ofoldx
COPY scripts ./scripts

# The build context intentionally excludes .git. Supply a deterministic version
# so setuptools-scm can build OFoldX from the copied source tree.
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OFOLDX="${OFOLDX_VERSION}" \
    python -m pip install . \
    && python -c "import torch; assert torch.version.cuda is not None; import ofoldx; print('OFoldX import OK; Torch', torch.__version__, 'CUDA', torch.version.cuda)"

CMD ["bash"]
