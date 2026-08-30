# ============================================================
# Contact-GraspNet PyTorch + ROS2 Humble
#
# Target environment:
#   Ubuntu 22.04
#   Python 3.10
#   ROS2 Humble
#   CUDA 12.1
#   PyTorch 2.1.1 + cu121
#   torchvision 0.16.1 + cu121
#   cgn-pytorch 0.4.3
#
# ============================================================

FROM docker.io/osrf/ros:humble-desktop-full
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    ca-certificates \
    git \
    wget \
    python3-pip \
    python3-dev \
    python3-colcon-common-extensions \
    ros-humble-sensor-msgs-py \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA CUDA repository
RUN wget -q \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb

# CUDA 12.1 runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-12-1 \
    cuda-nvtx-12-1 \
    libcublas-12-1 \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda-12.1/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN pip3 install --no-cache-dir --upgrade pip

# ------------------------------------------------------------
# Install the exact PyTorch version expected by CGN
#
# cgn-pytorch 0.4.3 requires:
#
#     torch==2.1.1
#
# PyTorch officially provides the 2.1.1 + CUDA 12.1 wheel.
# ------------------------------------------------------------
RUN pip3 install --no-cache-dir \
    torch==2.1.1 \
    torchvision==0.16.1 \
    --index-url https://download.pytorch.org/whl/cu121


# ------------------------------------------------------------
# Install CGN
#
# --no-deps is intentional.
#
# cgn-pytorch has a number of tightly pinned dependencies.
# We install those explicitly below rather than allowing pip
# to silently modify the PyTorch environment.
# ------------------------------------------------------------
RUN pip3 install --no-cache-dir \
    cgn-pytorch==0.4.3 \
    --no-build-isolation \
    --no-deps

# ------------------------------------------------------------
# Install CGN's pinned Python dependencies
# ------------------------------------------------------------
RUN pip3 install --no-cache-dir --ignore-installed \
    asttokens==2.4.1 \
    certifi==2023.11.17 \
    charset-normalizer==3.3.2 \
    decorator==5.1.1 \
    exceptiongroup==1.2.0 \
    executing==2.0.1 \
    filelock==3.13.1 \
    fsspec==2023.10.0 \
    idna==3.4 \
    importlib-resources==6.1.1 \
    ipython==8.17.2 \
    jedi==0.19.1 \
    jinja2==3.1.2 \
    joblib==1.3.2 \
    markupsafe==2.1.3 \
    matplotlib-inline==0.1.6 \
    meshcat \
    mpmath==1.3.0 \
    networkx==3.2.1 \
    numpy==1.26.2 \
    parso==0.8.3 \
    pexpect==4.8.0 \
    pillow==10.1.0 \
    prompt-toolkit==3.0.41 \
    psutil==5.9.6 \
    ptyprocess \
    pure-eval==0.2.2 \
    pygments==2.17.2 \
    pyngrok==7.0.1 \
    pyparsing==3.1.1 \
    pyrender \
    pyyaml==6.0.1 \
    pyzmq==25.1.1 \
    requests==2.31.0 \
    scikit-learn==1.3.2 \
    scipy==1.11.4 \
    six==1.16.0 \
    stack-data \
    sympy==1.12 \
    threadpoolctl==3.2.0 \
    tornado==6.3.3 \
    tqdm==4.66.1 \
    traitlets==5.13.0 \
    trimesh==4.0.4 \
    typeguard==4.1.5 \
    typing-extensions==4.8.0 \
    u-msgpack-python \
    urllib3==2.1.0 \
    wcwidth==0.2.12

# ------------------------------------------------------------
# Install PyTorch Geometric
#
# IMPORTANT:
# These MUST be compiled for the same PyTorch ABI.
#
# We are using:
#     torch 2.1.1 + CUDA 12.1
# ------------------------------------------------------------
RUN pip3 install --no-cache-dir \
    torch-geometric==2.4.0

# ------------------------------------------------------------
# Install PyG compiled extensions
#
# If this doesn't work then we will see:
#     _ZN5torch3jit17parseSchemaOrNameERKSsb
#
# they are built for PyTorch 2.1.
# ------------------------------------------------------------
RUN pip3 install --no-cache-dir \
    pyg_lib \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# TBD
RUN pip3 install --no-cache-dir --force-reinstall \
    certifi==2023.11.17 \
    charset-normalizer==3.3.2 \
    requests==2.31.0 \
    typing-extensions==4.8.0 \
    urllib3==2.1.0 \
    idna==3.4

# ------------------------------------------------------------
# Test cases
# ------------------------------------------------------------
RUN python3 -c "\
import torch; \
print('========================================'); \
print('PyTorch:', torch.__version__); \
print('CUDA:', torch.version.cuda); \
print('CUDA available:', torch.cuda.is_available()); \
print('========================================')"

RUN python3 -c "\
import torch; \
import torch_geometric; \
import torch_scatter; \
import torch_cluster; \
import torch_sparse; \
import torch_spline_conv; \
from cgn_pytorch import CGN; \
print('========================================'); \
print('ALL CGN DEPENDENCIES OK'); \
print('PyTorch:', torch.__version__); \
print('PyG:', torch_geometric.__version__); \
print('PyG scatter:', torch_scatter.__version__); \
print('PyG cluster:', torch_cluster.__version__); \
print('PyG sparse:', torch_sparse.__version__); \
print('PyG spline:', torch_spline_conv.__version__); \
print('CGN: OK'); \
print('========================================')"

# UOIS installs
RUN pip3 install --no-cache-dir --force-reinstall pypng
RUN pip3 install --no-deps scikit-image==0.23.2

RUN pip3 check

WORKDIR /cgn_ros2_ws
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]