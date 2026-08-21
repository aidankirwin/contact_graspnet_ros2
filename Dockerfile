# Use ROS2 Humble Base with Ubuntu 22.04 (Matches Python 3.10 perfectly)
FROM docker.io/osrf/ros:humble-desktop-full

# Minimize terminal interactions during build
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install Essential System Tools, Network Utilities, and Python 3.10 Dev libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    ca-certificates \
    git \
    wget \
    python3-pip \
    python3-dev \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# 2. Safely bootstrap official NVIDIA CUDA repository config and signing keys via network pin
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && sudo dpkg -i cuda-keyring_1.1-1_all.deb \
    && sudo rm cuda-keyring_1.1-1_all.deb

# 3. Install CUDA Toolkit runtime libraries (Required for PyTorch GPU execution)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-12-1 \
    cuda-nvtx-12-1 \
    libcublas-12-1 \
    libcudnn9-cuda-12 \
    ros-humble-sensor-msgs-py \
    && rm -rf /var/lib/apt/lists/*

# Set up environment variables for NVIDIA Runtime recognition
ENV PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# 4. Install CUDA-compatible PyTorch and the Contact-GraspNet port
# ADDED --ignore-installed to bypass the sympy distutils block
RUN pip3 install --upgrade pip && \
    pip3 install --ignore-installed torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# Install Contact-GraspNet bypassing build isolation
RUN pip3 install cgn-pytorch --no-build-isolation

# 5. Set up ROS2 Entrypoint workspace
WORKDIR /cgn_ros2_ws
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Default entrypoint sourcing ROS2 pathways automatically
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
