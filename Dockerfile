FROM docker.io/library/ros:humble-ros-base

# ============================================================
# Environment
# ============================================================

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    ROS_DISTRO=humble

ARG USER_UID=1001
ARG USER_GID=1001
ARG USERNAME=user

WORKDIR /cgn_ros2_ws

# ============================================================
# Basic development tools
# ============================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash-completion \
        curl \
        git \
        nano \
        vim \
        gdb \
        sudo \
        iputils-ping \
        openssh-client \
        python3-pip \
        python3-colcon-common-extensions \
        python3-colcon-argcomplete \
        libudev-dev \
        udev \
        usbutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# X11 / OpenGL
# ============================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        xauth \
        x11-apps \
        mesa-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*s


# ============================================================
# ROS 2 Python / GUI dependencies
# ============================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ros-humble-ros-gz \
        ros-humble-ros-gz-interfaces \
        ros-humble-rclpy \
        ros-humble-std-msgs \
        ros-humble-geometry-msgs \
        ros-humble-sensor-msgs \
        ros-humble-visualization-msgs \
        ros-humble-tf2 \
        ros-humble-tf2-ros \
        ros-humble-tf2-geometry-msgs \
        ros-humble-moveit \
        ros-humble-moveit-ros-move-group \
        ros-humble-moveit-kinematics \
        ros-humble-moveit-planners-ompl \
        ros-humble-moveit-ros-visualization \
        ros-humble-moveit-simple-controller-manager \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Docker CLI (hopefully temporary, I'm undecided about the DiD model)
# ============================================================
RUN apt-get update && apt-get install -y \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# User configuration
# ============================================================

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && mkdir -p -m 0700 /run/user/"${USER_UID}" \
    && mkdir -p -m 0700 /run/user/"${USER_UID}"/gdm \
    && chown -R $USERNAME:$USERNAME /run/user/"${USER_UID}" \
    && chown $USERNAME:$USERNAME /cgn_ros2_ws

ENV XDG_RUNTIME_DIR=/run/user/"${USER_UID}"
RUN echo "user soft rtprio 99" >> /etc/security/limits.conf
RUN echo "user hard rtprio 99" >> /etc/security/limits.conf
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> /home/$USERNAME/.bashrc

# ============================================================
# Workspace
# ============================================================

USER root
COPY . /cgn_ros2_ws/src
RUN sudo chown -R $USERNAME:$USERNAME /cgn_ros2_ws

SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash"]
WORKDIR /cgn_ros2_ws