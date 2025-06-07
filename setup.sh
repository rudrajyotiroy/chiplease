# Starting steps
sudo apt-get install  -y wget build-essential xutils-dev bison zlib1g-dev flex \
      libglu1-mesa-dev git g++ libssl-dev libxml2-dev libboost-all-dev git g++ \
      libxml2-dev vim python-setuptools build-essential python3-pip

pip3 install pyyaml plotly psutil
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sh cuda_12.8.1_570.124.06_linux.run --silent --toolkit
rm cuda_12.8.1_570.124.06_linux.run

export CUDA_INSTALL_PATH=/usr/local/cuda-11.0
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
./util/tracer_nvbit/install_nvbit.sh
make -C ./util/tracer_nvbit/

# # Make sure CUDA_INSTALL_PATH is set, and PATH includes nvcc
# # Get the applications, their data files and build them:
# git clone https://github.com/accel-sim/gpu-app-collection
# source ./gpu-app-collection/src/setup_environment
# make -j -C ./gpu-app-collection/src rodinia_2.0-ft
# make -C ./gpu-app-collection/src data

pip3 install -r requirements.txt

./rebuild.sh