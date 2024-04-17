rm -rf build && mkdir build && cd build && cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ../ && cmake --build . && ./gpu
