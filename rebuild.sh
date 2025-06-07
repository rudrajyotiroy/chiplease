source ./gpu-simulator/setup_environment.sh

# Build with make
make -j -C ./gpu-simulator/

# Build with CMake
cmake -S ./gpu-simulator/ -B ./gpu-simulator/build
cmake --build ./gpu-simulator/build -j8
cmake --install ./gpu-simulator/build

./gpu-simulator/bin/release/accel-sim.out -trace ./hw_run/ubench/11.0/l1_bw_32f_unroll_large/NO_ARGS/traces/kernelslist.g -config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/gpgpusim.config -config ./gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config