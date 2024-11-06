########################################################################################################################
# This is a fork of: https://github.com/huggingface/text-generation-inference/blob/v2.4.0/Dockerfile
# and based on code from: https://github.com/IBM/text-generation-inference/blob/8a8c55dd/Dockerfile
# and from: https://github.com/LukeMathWalker/cargo-chef/tree/v0.1.68?tab=readme-ov-file#without-the-pre-built-image
# and from: https://github.com/rust-lang/docker-rust/blob/1700955/stable/bullseye/Dockerfile
# So, that we can have a redhat base image, but the huggingface functionality
########################################################################################################################

ARG BASE_UBI_IMAGE_TAG=9.4-1214

# Rust builder
FROM registry.access.redhat.com/ubi9/ubi:${BASE_UBI_IMAGE_TAG} AS chef

RUN dnf install -y wget gcc g++; \
    dnf clean all

# adapted from: https://github.com/rust-lang/docker-rust/blob/1700955/stable/bullseye/Dockerfile
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_VERSION=1.82.0

RUN set -eux; \
    dpkgArch="$(arch)"; \
    case "${dpkgArch##*-}" in \
        x86_64) rustArch='x86_64-unknown-linux-gnu'; rustupSha256='6aeece6993e902708983b209d04c0d1dbb14ebb405ddb87def578d41f920f56d' ;; \
        armv7) rustArch='armv7-unknown-linux-gnueabihf'; rustupSha256='3c4114923305f1cd3b96ce3454e9e549ad4aa7c07c03aec73d1a785e98388bed' ;; \
        aarch64) rustArch='aarch64-unknown-linux-gnu'; rustupSha256='1cffbf51e63e634c746f741de50649bbbcbd9dbe1de363c9ecef64e278dba2b2' ;; \
        i386) rustArch='i686-unknown-linux-gnu'; rustupSha256='0a6bed6e9f21192a51f83977716466895706059afb880500ff1d0e751ada5237' ;; \
        *) echo >&2 "unsupported architecture: ${dpkgArch}"; exit 1 ;; \
    esac; \
    url="https://static.rust-lang.org/rustup/archive/1.27.1/${rustArch}/rustup-init"; \
    wget "$url"; \
    echo "${rustupSha256} *rustup-init" | sha256sum -c -; \
    chmod +x rustup-init; \
    ./rustup-init -y --no-modify-path --profile minimal --default-toolchain $RUST_VERSION --default-host ${rustArch}; \
    rm rustup-init; \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME; \
    rustup --version; \
    cargo --version; \
    rustc --version

RUN cargo install cargo-chef

WORKDIR /usr/src

ARG CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

FROM chef AS planner
COPY Cargo.lock Cargo.lock
COPY Cargo.toml Cargo.toml
COPY rust-toolchain.toml rust-toolchain.toml
COPY proto proto
COPY benchmark benchmark
COPY router router
COPY backends backends
COPY launcher launcher

RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

# installing both 3.9 and 3.11, but setting 3.11 as default (cannot clear cargo attempting to link to 3.9).
RUN dnf install -y python3.11-devel python3.9-devel unzip openssl-devel; \
    alternatives --install /usr/bin/python python /usr/bin/python3.11 1; \
    dnf clean all

RUN PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP

COPY --from=planner /usr/src/recipe.json recipe.json
RUN cargo chef cook --profile release-opt --recipe-path recipe.json

ARG GIT_SHA
ARG DOCKER_LABEL

COPY Cargo.lock Cargo.lock
COPY Cargo.toml Cargo.toml
COPY rust-toolchain.toml rust-toolchain.toml
COPY proto proto
COPY benchmark benchmark
COPY router router
COPY backends backends
COPY launcher launcher
RUN cargo build --profile release-opt --frozen

# Python builder
# Adapted from: https://github.com/pytorch/pytorch/blob/master/Dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubi9 AS pytorch-install

# NOTE: When updating PyTorch version, beware to remove `pip install nvidia-nccl-cu12==2.22.3` below in the Dockerfile. Context: https://github.com/huggingface/text-generation-inference/pull/2099
ARG PYTORCH_VERSION=2.4.0

ARG PYTHON_VERSION=3.11
# Keep in sync with `server/pyproject.toml
ARG CUDA_VERSION=12.4
ARG MAMBA_VERSION=24.3.0-0
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch
# Automatically set by buildx
ARG TARGETPLATFORM

ENV PATH /opt/conda/bin:$PATH

RUN dnf install -y \
        ca-certificates \
        git; \
    dnf clean all

# Install conda
# translating Docker's TARGETPLATFORM into mamba arches
RUN case ${TARGETPLATFORM} in \
         "linux/arm64")  MAMBA_ARCH=aarch64  ;; \
         *)              MAMBA_ARCH=x86_64   ;; \
    esac && \
    curl -fsSL -v -o ~/mambaforge.sh -O  "https://github.com/conda-forge/miniforge/releases/download/${MAMBA_VERSION}/Mambaforge-${MAMBA_VERSION}-Linux-${MAMBA_ARCH}.sh"
RUN chmod +x ~/mambaforge.sh && \
    bash ~/mambaforge.sh -b -p /opt/conda && \
    rm ~/mambaforge.sh

# Install pytorch
# On arm64 we exit with an error code
RUN case ${TARGETPLATFORM} in \
         "linux/arm64")  exit 1 ;; \
         *)              /opt/conda/bin/conda update -y conda &&  \
                         /opt/conda/bin/conda install -c "${INSTALL_CHANNEL}" -c "${CUDA_CHANNEL}" -y "python=${PYTHON_VERSION}" "pytorch=$PYTORCH_VERSION" "pytorch-cuda=$(echo $CUDA_VERSION | cut -d'.' -f 1-2)"  ;; \
    esac && \
    /opt/conda/bin/conda clean -ya

# CUDA kernels builder image
FROM pytorch-install AS kernel-builder

ARG MAX_JOBS=8
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0+PTX"

RUN dnf install -y ninja-build cmake; \
    dnf clean all

# Build Flash Attention CUDA kernels
FROM kernel-builder AS flash-att-builder

WORKDIR /usr/src

COPY server/Makefile-flash-att Makefile

# Build specific version of flash attention
RUN make build-flash-attention

# Build Flash Attention v2 CUDA kernels
FROM kernel-builder AS flash-att-v2-builder

WORKDIR /usr/src

COPY server/Makefile-flash-att-v2 Makefile

# Build specific version of flash attention v2
RUN make build-flash-attention-v2-cuda

# Build Transformers exllama kernels
FROM kernel-builder AS exllama-kernels-builder
WORKDIR /usr/src
COPY server/exllama_kernels/ .

RUN python setup.py build

# Build Transformers exllama kernels
FROM kernel-builder AS exllamav2-kernels-builder
WORKDIR /usr/src
COPY server/Makefile-exllamav2/ Makefile

# Build specific version of transformers
RUN make build-exllamav2

# Build Transformers awq kernels
FROM kernel-builder AS awq-kernels-builder
WORKDIR /usr/src
COPY server/Makefile-awq Makefile
# Build specific version of transformers
RUN make build-awq

# Build eetq kernels
FROM kernel-builder AS eetq-kernels-builder
WORKDIR /usr/src
COPY server/Makefile-eetq Makefile
# Build specific version of transformers
RUN make build-eetq

# Build Lorax Punica kernels
FROM kernel-builder AS lorax-punica-builder
WORKDIR /usr/src
COPY server/Makefile-lorax-punica Makefile
# Build specific version of transformers
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" make build-lorax-punica

# Build Transformers CUDA kernels
FROM kernel-builder AS custom-kernels-builder
WORKDIR /usr/src
COPY server/custom_kernels/ .
# Build specific version of transformers
RUN python setup.py build

# Build vllm CUDA kernels
FROM kernel-builder AS vllm-builder

WORKDIR /usr/src

ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"

COPY server/Makefile-vllm Makefile

# Build specific version of vllm
RUN make build-vllm-cuda

# Build mamba kernels
FROM kernel-builder AS mamba-builder
WORKDIR /usr/src
COPY server/Makefile-selective-scan Makefile
RUN make build-all

# Build flashinfer
FROM kernel-builder AS flashinfer-builder
WORKDIR /usr/src
COPY server/Makefile-flashinfer Makefile
RUN make install-flashinfer


# Base Layer ##################################################################
FROM registry.access.redhat.com/ubi9/ubi:${BASE_UBI_IMAGE_TAG} AS base
WORKDIR /app

ARG PYTHON_VERSION

RUN dnf remove -y --disableplugin=subscription-manager \
        subscription-manager \
        # we install newer version of requests via pip
    python${PYTHON_VERSION}-requests \
    && dnf install -y make \
        # to help with debugging
        procps \
    && dnf clean all

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

FROM base AS cuda-base

# Ref: https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/
ENV CUDA_VERSION=12.4.1 \
    NV_CUDA_LIB_VERSION=12.4.1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NV_CUDA_CUDART_VERSION=12.4.127 \
    NV_CUDA_COMPAT_VERSION=550.54.15

RUN dnf config-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
    && dnf install -y \
        cuda-cudart-12-4-${NV_CUDA_CUDART_VERSION} \
        cuda-compat-12-4-${NV_CUDA_COMPAT_VERSION} \
    && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf \
    && dnf clean all

ENV CUDA_HOME="/usr/local/cuda" \
    PATH="/usr/local/nvidia/bin:${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

#FROM cuda-base as cuda-devel
#
## Ref: https://developer.nvidia.com/nccl/nccl-legacy-downloads
#ENV NV_CUDA_CUDART_DEV_VERSION=12.1.55-1 \
#    NV_NVML_DEV_VERSION=12.1.55-1 \
#    NV_LIBCUBLAS_DEV_VERSION=12.1.0.26-1 \
#    NV_LIBNPP_DEV_VERSION=12.0.2.50-1 \
#    NV_LIBNCCL_DEV_PACKAGE_VERSION=2.18.3-1+cuda12.1
#
#RUN dnf config-manager \
#       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
#    && dnf install -y \
#        cuda-command-line-tools-12-1-${NV_CUDA_LIB_VERSION} \
#        cuda-libraries-devel-12-1-${NV_CUDA_LIB_VERSION} \
#        cuda-minimal-build-12-1-${NV_CUDA_LIB_VERSION} \
#        cuda-cudart-devel-12-1-${NV_CUDA_CUDART_DEV_VERSION} \
#        cuda-nvml-devel-12-1-${NV_NVML_DEV_VERSION} \
#        libcublas-devel-12-1-${NV_LIBCUBLAS_DEV_VERSION} \
#        libnpp-devel-12-1-${NV_LIBNPP_DEV_VERSION} \
#        libnccl-devel-${NV_LIBNCCL_DEV_PACKAGE_VERSION} \
#    && dnf clean all
#
#ENV LIBRARY_PATH="$CUDA_HOME/lib64/stubs"

FROM cuda-base AS inference-base

# Conda env
ENV PATH=/opt/conda/bin:$PATH \
    CONDA_PREFIX=/opt/conda

# Text Generation Inference base env
ENV HF_HOME=/data \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=80

WORKDIR /usr/src

RUN dnf install -y \
        openssl-devel \
        ca-certificates \
        make \
        git; \
    dnf clean all

# Copy conda with PyTorch installed
COPY --from=pytorch-install /opt/conda /opt/conda

# Copy build artifacts from flash attention builder
COPY --from=flash-att-builder /usr/src/flash-attention/build/lib.linux-x86_64-cpython-311 /opt/conda/lib/python3.11/site-packages
COPY --from=flash-att-builder /usr/src/flash-attention/csrc/layer_norm/build/lib.linux-x86_64-cpython-311 /opt/conda/lib/python3.11/site-packages
COPY --from=flash-att-builder /usr/src/flash-attention/csrc/rotary/build/lib.linux-x86_64-cpython-311 /opt/conda/lib/python3.11/site-packages

# Copy build artifacts from flash attention v2 builder
COPY --from=flash-att-v2-builder /opt/conda/lib/python3.11/site-packages/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so /opt/conda/lib/python3.11/site-packages

# Copy build artifacts from custom kernels builder
COPY --from=custom-kernels-builder /usr/src/build/lib.linux-x86_64-cpython-311 /opt/conda/lib/python3.11/site-packages
# Copy build artifacts from exllama kernels builder
COPY --from=exllama-kernels-builder /usr/src/build/lib.linux-x86_64-cpython-311 /opt/conda/lib/python3.11/site-packages
# Copy build artifacts from exllamav2 kernels builder
COPY --from=exllamav2-kernels-builder /usr/src/exllamav2/build/lib.linux-x86_64-cpython-311 /opt/conda/lib/python3.11/site-packages
# Copy build artifacts from awq kernels builder
COPY --from=awq-kernels-builder /usr/src/llm-awq/awq/kernels/build/lib.linux-x86_64-cpython-311 /opt/conda/lib/python3.11/site-packages
# Copy build artifacts from eetq kernels builder
COPY --from=eetq-kernels-builder /usr/src/eetq/build/lib.linux-x86_64-cpython-311 /opt/conda/lib/python3.11/site-packages
# Copy build artifacts from lorax punica kernels builder
COPY --from=lorax-punica-builder /usr/src/lorax-punica/server/punica_kernels/build/lib.linux-x86_64-cpython-311 /opt/conda/lib/python3.11/site-packages
# Copy build artifacts from vllm builder
COPY --from=vllm-builder /usr/src/vllm/build/lib.linux-x86_64-cpython-311 /opt/conda/lib/python3.11/site-packages
# Copy build artifacts from mamba builder
COPY --from=mamba-builder /usr/src/mamba/build/lib.linux-x86_64-cpython-311/ /opt/conda/lib/python3.11/site-packages
COPY --from=mamba-builder /usr/src/causal-conv1d/build/lib.linux-x86_64-cpython-311/ /opt/conda/lib/python3.11/site-packages
COPY --from=flashinfer-builder /opt/conda/lib/python3.11/site-packages/flashinfer/ /opt/conda/lib/python3.11/site-packages/flashinfer/

# Install flash-attention dependencies
RUN pip install einops --no-cache-dir

# Install server
COPY proto proto
COPY server server
COPY server/Makefile server/Makefile
RUN cd server && \
    make gen-server && \
    pip install -r requirements_cuda.txt && \
    pip install ".[bnb, accelerate, marlin, moe, quantize, peft, outlines]" --no-cache-dir && \
    pip install nvidia-nccl-cu12==2.22.3

ENV LD_PRELOAD=/opt/conda/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2
# Required to find libpython within the rust binaries
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/conda/lib/"
# This is needed because exl2 tries to load flash-attn
# And fails with our builds.
ENV EXLLAMA_NO_FLASH_ATTN=1

# Deps before the binaries
# The binaries change on every build given we burn the SHA into them
# The deps change less often.
RUN dnf install -y g++; \
    dnf clean all

# Install benchmarker
COPY --from=builder /usr/src/target/release-opt/text-generation-benchmark /usr/local/bin/text-generation-benchmark
# Install router
COPY --from=builder /usr/src/target/release-opt/text-generation-router /usr/local/bin/text-generation-router
# Install launcher
COPY --from=builder /usr/src/target/release-opt/text-generation-launcher /usr/local/bin/text-generation-launcher


# AWS Sagemaker compatible image
FROM inference-base AS sagemaker

COPY sagemaker-entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

# Final image
FROM inference-base

COPY ./tgi-entrypoint.sh /tgi-entrypoint.sh
RUN chmod +x /tgi-entrypoint.sh

ENTRYPOINT ["/tgi-entrypoint.sh"]
# CMD ["--json-output"]
