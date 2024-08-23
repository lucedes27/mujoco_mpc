# Use the official Ubuntu 20.04 image as the base
FROM ubuntu:20.04

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install required dependencies
RUN apt-get update && apt-get -y install \
    libgl1-mesa-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxrandr-dev \
    libxi-dev \
    ninja-build \ 
    zlib1g-dev \
    clang-12 \
    clang++-12 \
    cmake \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /mujoco_mpc

# Clone the MuJoCo repository (you can replace this with your repository if needed)
# RUN git clone https://github.com/google-deepmind/mujoco_mpc.git .
COPY . .

# Set the MUJOCO_MPC_MUJOCO_GIT_TAG environment variable (replace 'main' with the desired tag/branch)
ARG MUJOCO_MPC_MUJOCO_GIT_TAG=main
ENV MUJOCO_MPC_MUJOCO_GIT_TAG=${MUJOCO_MPC_MUJOCO_GIT_TAG}

WORKDIR /mujoco_mpc/build

# Configure MuJoCo MPC
RUN cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang-12 \
    -DCMAKE_CXX_COMPILER=clang++-12 \
    -DMJPC_BUILD_GRPC_SERVICE=ON && \
    cmake --build . --config Release

WORKDIR /mujoco_mpc

RUN apt-get update && apt-get install -y \
    software-properties-common curl && \
    add-apt-repository -y ppa:deadsnakes/ppa  && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-dev python3-apt && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# RUN pip install -r requirements.txt

WORKDIR /mujoco_mpc/python

RUN pip install --ignore-installed PyYAML==6.0 && python3.10 setup.py install

RUN pip install torch matplotlib

# Run tests (this is optional, depending on whether you want to run tests as part of the build)
# CMD cd mjpc/test && ctest -C Release --output-on-failure .
CMD /bin/bash
