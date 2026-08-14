# TFM4GAD environment for Blackwell GPUs (RTX 5060/5070/5080/5090, sm_120)
#
# Adapted from JMB-Scripts/RFdiffusion-dockerfile-nvidia-RTX5090
# https://github.com/JMB-Scripts/RFdiffusion-dockerfile-nvidia-RTX5090
# Core recipe (CUDA 12.8 + PyTorch nightly cu128 + DGL compiled from
# source against that PyTorch build) kept as close to the original as
# possible, since that combination is the one with a verified working
# report on Blackwell hardware.
#
# CHANGES from the reference:
#   - Removed all RFdiffusion/SE3Transformer-specific build steps.
#   - The repo is NOT copied into the image (no `COPY . .`). Mount your
#     local TFM4GAD clone as a volume at `docker run` time, or via a
#     VS Code Dev Container, so editing code doesn't require a rebuild.
#   - Package list in step 8 is a placeholder. CONFIRM against TFM4GAD's
#     actual requirements/imports before relying on this - see the note
#     at the top of the chat response about verifying whether `dgl` is
#     even imported anywhere in the repo. If it isn't, you can delete
#     step 7 entirely and save a large chunk of build time.
#
# NOT tested by Claude in a live build - there's no GPU and no route to
# NVIDIA's CUDA repo domain in the sandbox this was written in. Treat
# this as a documented starting point, not a verified artifact.

#-----------------------------------------------------------------------
# STAGE 1: BUILDER
#-----------------------------------------------------------------------
FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

# 1. System dependencies
RUN apt-get -q update && \
    apt-get install -y --no-install-recommends \
        git wget curl ca-certificates gnupg \
        build-essential ninja-build g++-11 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Modern CMake (Ubuntu 22.04's default 3.22 is too old to build DGL)
ENV CMAKE_VERSION=3.29.3
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && \
    tar -xzf cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz --strip-components=1 -C /usr/local && \
    rm cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz

# 3. CUDA Toolkit 12.8 - first CUDA release with sm_120 (Blackwell) support
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get -q update && \
    apt-get -y install cuda-toolkit-12-8 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    rm cuda-keyring_1.1-1_all.deb

ENV PATH="/usr/local/cuda-12.8/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64"
ENV CUDA_HOME="/usr/local/cuda-12.8"
ENV CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda-12.8"
ENV CUDA_PATH="/usr/local/cuda-12.8"

# 4. Mambaforge (conda env manager)
ENV CONDA_DIR=/opt/conda
RUN wget --quiet "https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Mambaforge-24.3.0-0-Linux-x86_64.sh" -O ~/mambaforge.sh && \
    /bin/bash ~/mambaforge.sh -b -p /opt/conda && \
    rm ~/mambaforge.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# 5. Conda environment
#    Adjust python version if TFM4GAD's own docs/imports need a
#    different one - not confirmed yet.
RUN mamba create -n tfm4gad python=3.11 -y
SHELL ["conda", "run", "-n", "tfm4gad", "/bin/bash", "-c"]

# 6. PyTorch stable for CUDA 12.8 - install FIRST, DGL builds against it
RUN pip install --no-cache-dir torch==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 7. Compile DGL from source against the stable PyTorch - CPU-ONLY build
#    DGL is required: it IS imported by TFM4GAD (dataloader.py and
#    augment.py). But it is only used for preprocessing on CPU:
#      - dataloader.py: load_graphs() + to_bidirected/add_self_loop
#      - augment.py:    message passing (update_all) for SAGE/BWGNN,
#                       degree encoding, PageRank, and Laplacian PE
#    The graph is never moved to GPU - there is no .to('cuda'),
#    graph.to(device), or dgl.to_device call anywhere in the repo. Features
#    cross to the GPU only as numpy arrays handed to TabPFN, which runs on
#    GPU through PyTorch. So USE_CUDA=ON would buy no speedup here and only
#    makes this step far heavier to compile.
WORKDIR /tmp
RUN git clone --recursive https://github.com/dmlc/dgl.git && \
    cd dgl && mkdir build && cd build && \
    cmake -D USE_CUDA=OFF -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 \
          -D BUILD_TYPE=release -D CMAKE_BUILD_TYPE=Release \
          -D BUILD_CPP_TEST=OFF .. && \
    make -j1 && \
    cd ../python && pip install .

# 8. TFM4GAD / GADBench dependencies
#    PLACEHOLDER LIST - replace with exact pins once you've inspected
#    the repo's actual requirements/environment file with Claude Code.
RUN pip install --no-cache-dir \
    tabpfn numpy pandas scipy scikit-learn networkx pyyaml tqdm hydra-core

#-----------------------------------------------------------------------
# STAGE 2: FINAL IMAGE (lean runtime, no build tools)
#-----------------------------------------------------------------------
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# CUDA compat package only - PyTorch's pip wheel brings its own CUDA
# runtime libs. This package bridges to the host driver.
RUN apt-get -q update && \
    apt-get install -y --no-install-recommends curl ca-certificates gnupg wget git && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get -q update && \
    apt-get -y install cuda-compat-12-8 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    rm cuda-keyring_1.1-1_all.deb

ENV CONDA_DIR=/opt/conda
COPY --from=builder /opt/conda /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

ENV DGLBACKEND="pytorch"
ENV PYTHONUNBUFFERED=1
# Deliberately NOT setting LD_LIBRARY_PATH here - conflicts with
# PyTorch's bundled CUDA libs (same reasoning as the reference Dockerfile).

WORKDIR /workspace
# Your local TFM4GAD clone is bind-mounted here at `docker run` time -
# see the run command / devcontainer.json in the chat response.

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tfm4gad"]
CMD ["bash"]
