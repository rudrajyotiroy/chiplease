export CUDA_INSTALL_PATH=/usr/local/cuda-11.0
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
./util/tracer_nvbit/install_nvbit.sh
make -C ./util/tracer_nvbit/

# Make sure CUDA_INSTALL_PATH is set, and PATH includes nvcc
# Get the applications, their data files and build them:
git clone https://github.com/accel-sim/gpu-app-collection
source ./gpu-app-collection/src/setup_environment
make -j -C ./gpu-app-collection/src rodinia_2.0-ft
make -C ./gpu-app-collection/src data

pip3 install -r requirements.txt
source ./gpu-simulator/setup_environment.sh

# Build with make
make -j -C ./gpu-simulator/

# Build with CMake
cmake -S ./gpu-simulator/ -B ./gpu-simulator/build
cmake --build ./gpu-simulator/build -j8
cmake --install ./gpu-simulator/build

./gpu-simulator/bin/release/accel-sim.out -trace ./hw_run/rodinia_2.0-ft/9.1/backprop-rodinia-2.0-ft/4096___data_result_4096_txt/traces/kernelslist.g -config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/gpgpusim.config -config ./gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config