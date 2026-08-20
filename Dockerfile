# Use CUDA 11.8 devel on Ubuntu 22.04 (Compatible with both 2080Ti and 4090 hardware)
FROM docker.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

# Prevent interactive configuration screens
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install baseline OS packages, build utilities, and required GUI/X11 libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    software-properties-common \
    libgl1-mesa-glx \
    libegl1-mesa \
    libgles2-mesa \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxt6 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*


# Install complete ROS 2 Humble Desktop alongside build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Upgrade packaging managers
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install common unified Python ML / Geometric stack
RUN python3 -m pip install --no-cache-dir \
    tensorflow==2.12.0 \
    numpy==1.23.5 \
    scipy==1.10.1 \
    matplotlib==3.7.1 \
    trimesh==3.21.5 \
    h5py==3.8.0 \
    opencv-python==4.7.0.72 \
    pyyaml==6.0 \
    tqdm==4.65.0 \
    shapely==2.0.1 \
    pyrender==0.1.43 \
    python-fcl==0.0.12 \
    imageio==2.28.1


# CRITICAL STEP FOR MULTI-GPU MACHINES:
# Intercept and modify compile script to build universal binary payloads (Fat Binaries)
# Maps Turing (arch=sm_75) for 2080Ti and Ada Lovelace (arch=sm_89) for 4090.
WORKDIR /cgn_ws/contact_graspnet_ros2/contact_graspnet/pointnet2/tf_ops
RUN sed -i 's/-arch=sm_35/-gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_89,code=sm_89/g' compile_pointnet_tfops.sh && \
    chmod +x compile_pointnet_tfops.sh && \
    ./compile_pointnet_tfops.sh || true

# Global terminal environment variables hook setup
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

WORKDIR /cgn_ws
CMD ["bash"]
